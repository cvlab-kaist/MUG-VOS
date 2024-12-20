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

from torch.nn import functional as TF

# from model.losses_sam import LossComputer
from model.losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
from .utils.transforms import ResizeLongestSide

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
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.vid_sam_ddp.parameters()), lr=config['lr'],
            weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
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

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        ori_size = frames.shape[-2:]
        frames = self.sam_pad(frames)
        first_frame_gt = self.sam_pad(first_frame_gt)

        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        input_size = frames.shape[-2:]
        filled_embedding_size = (round(self.embedding_size / input_size[1] * input_size[0]), self.embedding_size)
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # image features never change, compute once
            key, shrinkage, selection, f16 = self.vid_sam_ddp('encode_key', frames)

            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            v16, hidden = self.vid_sam_ddp('encode_value', frames[:, 0], f16[:, 0], hidden, first_frame_gt[:, 0])
            values = v16.unsqueeze(3)  # add the time dimension

            for ti in range(1, self.num_frames):
                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:, :, :ti]
                    ref_shrinkage = shrinkage[:, :, :ti] if shrinkage is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would
                    # need broadcasting in gather which we don't have
                    indices = [
                        torch.cat([+filler_one, torch.randperm(ti - 1)[:self.num_ref_frames - 1] + 1])
                        for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None


                # Segment frame ti
                memory_readout = self.vid_sam_ddp('read_memory', key[:, :, ti],
                                                  selection[:, :, ti] if selection is not None else None,
                                                  ref_keys, ref_shrinkage, ref_values)
                hidden, logits, masks = self.vid_sam_ddp('segment', f16[:, ti], memory_readout,
                                                         hidden, selector, h_out=(ti < (self.num_frames - 1)))

                # No need to encode the last frame
                if ti < (self.num_frames - 1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.vid_sam_ddp('encode_value', frames[:, ti], f16[:, ti], hidden, masks,
                                                   is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

                masks = self.sam_unpad(masks, ori_size)
                logits = self.sam_unpad(logits, ori_size)

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:

                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time',
                                                (time.time() - self.last_time) / self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time',
                                                (time.time() - self.last_time) / self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            self.optimizer.step()

        self.scheduler.step()

    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.vid_sam_ddp.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = {
            'it': it,
            'network': self.vid_sam_ddp.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

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
        map_location = 'cuda:%d' % self.local_rank
        self.vid_sam_ddp.module.load_weights(torch.load(path))

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.vid_sam_ddp.train()
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