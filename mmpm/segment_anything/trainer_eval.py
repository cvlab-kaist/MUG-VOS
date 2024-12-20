"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from model.losses_sam import LossComputer
from model.losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
from util.result_saver import save_results
from .utils.transforms import ResizeLongestSide

from torch.nn import functional as TF


import cv2

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
# from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


class SamTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank

        self.vid_sam = sam_model_registry[config['sam_type']](
            single_object=config['single_object'], checkpoint=config['load_network'])

        self.vid_sam_ddp = nn.parallel.DistributedDataParallel(self.vid_sam.cuda(),
                                                               device_ids=[local_rank], output_device=local_rank,
                                                               broadcast_buffers=False)

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size',
                                   str(sum([param.nelement() for param in self.vid_sam_ddp.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.test()

        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

        self.transform = ResizeLongestSide(self.vid_sam.image_encoder.img_size)
        self.embedding_size = 64

        self.padding_size = 1024

        self.mem_interval = config['mem_interval']
        self.max_memory_length = config['max_memory_length']
        self.memory_filtering_method = config['memory_filtering_method']

        self.output_path = config['output_path']
        self.threshold = config['threshold']


    def do_eval(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        origin_size = tuple(i.item() for i in torch.cat(data['origin_size']))
        pallete = data['pallete']
        video_name = data['info']['name']


        padded_size = frames.shape[-2:]
        frames = self.sam_pad(frames)
        first_frame_gt = self.sam_pad(first_frame_gt)

        video_length = frames.shape[1]

        b = frames.shape[0] # [B, Frame_length, 3, H, W]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        input_size = frames.shape[-2:]
        filled_embedding_size = (round(self.embedding_size / input_size[1] * input_size[0]), self.embedding_size)
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            for ti in range(0, video_length):
                frame = frames[:, ti].unsqueeze(1)
                key, shrinkage, selection, f16 = self.vid_sam_ddp('encode_key', frame)

                if ti == 0:
                    hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
                    v16, hidden = self.vid_sam_ddp('encode_value', frames[:, 0], f16[:, 0], hidden,
                                                   first_frame_gt[:, 0])
                    values = v16.unsqueeze(3)  # add the time dimension

                    ref_keys = key
                    ref_shrinkage = shrinkage
                    ref_values = values
                else:
                    # Segment frame ti
                    memory_readout = self.vid_sam_ddp('read_memory', key[:, :, 0],
                                                      selection[:, :, 0] if selection is not None else None,
                                                      ref_keys, ref_shrinkage, ref_values)
                    hidden, logits, masks = self.vid_sam_ddp('segment', f16[:, 0], memory_readout,
                                                             hidden, selector, h_out=(ti < (video_length - 1)))

                    # No need to encode the last frame
                    if ti < (video_length - 1) and ti % int(self.mem_interval) == 0:
                        is_deep_update = np.random.rand() < self.deep_update_prob
                        v16, hidden = self.vid_sam_ddp('encode_value', frame[:, 0], f16[:, 0], hidden, masks,
                                                       is_deep_update=is_deep_update)

                        ref_keys = torch.cat([ref_keys, key], 2)
                        ref_shrinkage = torch.cat([ref_shrinkage, shrinkage], 2)
                        ref_values = torch.cat([ref_values, v16.unsqueeze(3)], 3)

                    masks = self.sam_unpad(masks, padded_size)
                    logits = self.sam_unpad(logits, padded_size)

                    out[f'masks_{ti}'] = masks
                    out[f'logits_{ti}'] = logits

                    ref_keys, ref_shrinkage, ref_values = self.filter_memory(ref_keys, ref_shrinkage, ref_values, self.memory_filtering_method)

            # Logging
            images = {**data, **out}

            output_path = os.path.join(self.output_path, str(self.threshold), video_name[0])
            save_results(images, origin_size, output_path, threshold=self.threshold, pallete=pallete)
            
            # for threshold in self.threshold:
            #     output_path = os.path.join(self.output_path, str(threshold), video_name[0])
            #     save_results(images, origin_size, output_path, threshold=threshold, pallete = pallete)
            # self.logger.log_cv2('train/pairs', save_results(images, size, num_filled_objects), it)

    def filter_memory(self, key, shrinkage, values, method="random_except_first_last"):
        """
        filtering Memory, Randomly sample memory with preserving first and last memory.
        :param key:
        :param shrinkage:
        :param values:
        :return:
        """
        if key.shape[2] > self.max_memory_length:
            first_index = 0
            last_index = key.shape[2]
            batch = key.shape[0]

            if method == "random_except_first_last":
                filter_one = torch.zeros(1, dtype=torch.int64)
                filter_last = torch.ones(1, dtype=torch.int64) * (last_index - 1)
                selected_indices = [
                            torch.sort(torch.cat([+filter_one, torch.randperm(last_index - 2)[:self.max_memory_length - 2] + 1, filter_last])).values for _ in range(batch)]

            elif method == "random_except_first":
                filter_one = torch.zeros(1, dtype=torch.int64)
                selected_indices = [
                    torch.cat([+filter_one, torch.arange(last_index-1)[:self.max_memory_length -1] + 1]) for _ in range(batch)]
            elif method == "random":
                selected_indices = [
                    torch.randperm(last_index)[:self.max_memory_length] for _ in range(batch)]
            elif method == "fifo":
                selected_indices = [torch.arange(last_index)[:self.max_memory_length] + 1 for _ in range(batch)]
            elif method == "not_filter":
                selected_indices = [torch.arange(last_index)[:self.max_memory_length] for _ in range(batch)]

            key = torch.stack([key[bi,:, selected_indices[bi]] for bi in range(batch)])
            shrinkage = torch.stack([shrinkage[bi,:, selected_indices[bi]] for bi in range(batch)])
            values = torch.stack([values[bi,:,:, selected_indices[bi]] for bi in range(batch)])

        return key, shrinkage, values


    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        if 'it' in checkpoint.keys():
            it = checkpoint['it']
            network = checkpoint['network']
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
            map_location = 'cuda:%d' % self.local_rank
            self.vid_sam_ddp.module.load_state_dict(network)
            self.optimizer.load_state_dict(optimizer)
            self.scheduler.load_state_dict(scheduler)
            print('Network weights, optimizer states, and scheduler states loaded.')
            return it
        else:
            self.vid_sam_ddp.module.load_state_dict(checkpoint)
            return 0

    def load_network(self, path):
        self.vid_sam_ddp.module.load_state_dict(torch.load(path))

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.vid_sam_ddp.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.vid_sam_ddp.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.vid_sam_ddp.eval()
        return self


    def sam_pad(self, x: torch.Tensor) -> torch.Tensor:
        # Pad
        h, w = x.shape[-2:]
        padh = self.padding_size - h
        padw = self.padding_size - w
        x = TF.pad(x, (0, padw, 0, padh))
        return x

    def sam_unpad(self, x: torch.Tensor, origin_size) -> torch.Tensor:
        padh = origin_size[0] - self.padding_size
        padw = origin_size[1] - self.padding_size
        x = TF.pad(x, (0, padw, 0, padh))
        return x