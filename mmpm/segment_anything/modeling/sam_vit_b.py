# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

import numpy as np
from ..utils.transforms import ResizeLongestSide

from model.aggregate import aggregate
from model.modules import KeyProjection, ValueEncoder, ValueEncoderViT

import cv2

import math
from typing import Optional

from functools import partial


class VidSam(nn.Module):
    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            mask_decoder: MaskDecoder,
            hidden_dim, key_dim, value_dim, single_object=False,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.single_object = single_object

        # Projection from f16 feature space to key/value space
        self.key_proj = KeyProjection(256, self.key_dim)  # Original 1024 -> VIT: 256

        # self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)
        self.input_channel = 4 if single_object else 5
        value_encoder_vit = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
            # in_chans=4 if single_object else 5,  # Image + Mask
            in_chans=4 if single_object else 5,  # Image + Mask
        )  # VIT -B SAM Version
        print("Value Channel 4") if single_object else print("Value Channel 5")
        self.pretrained_weight_path = "/media/data3/video_seg/trained_model/sam/sam_vit_b_01ec64.pth"
        self.value_encoder = ValueEncoderViT(value_encoder_vit, self.value_dim, self.hidden_dim, self.single_object)

        self.load_value_encoder_pretrained_weights()
        self._freeze_module(self.image_encoder)

    @torch.no_grad
    def image_encode(self, frames):
        input_frames = self.preprocess(frames)

        assert input_frames.shape[-2:] == (1024, 1024), f'Invalid Shape {input_frames.shape}'

        return self.image_encoder(input_frames)

    def read_memory(self, query_key, query_selection, memory_key,
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        similarity = self._get_similarity(memory_key, memory_shrinkage, query_key, query_selection)
        affinity = self._do_softmax(similarity)
        memory = self._readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def _get_similarity(self, mk, ms, qk, qe):
        # used for training/inference and memory reading/memory potentiation
        # mk: B x CK x [N]    - Memory keys
        # ms: B x  1 x [N]    - Memory shrinkage
        # qk: B x CK x [HW/P] - Query keys
        # qe: B x CK x [HW/P] - Query selection
        # Dimensions in [] are flattened
        CK = mk.shape[1]
        mk = mk.flatten(start_dim=2)
        ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
        qk = qk.flatten(start_dim=2)
        qe = qe.flatten(start_dim=2) if qe is not None else None

        if qe is not None:
            # See appendix for derivation
            # or you can just trust me ヽ(ー_ー )ノ
            mk = mk.transpose(1, 2)
            a_sq = (mk.pow(2) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            similarity = (-a_sq + two_ab - b_sq)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq + two_ab)

        if ms is not None:
            similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
        else:
            similarity = similarity / math.sqrt(CK)  # B*N*HW

        return similarity

    def _do_softmax(self, similarity, top_k: Optional[int] = None, inplace=False, return_usage=False):
        # normalize similarity with top-k softmax
        # similarity: B x N x [HW/P]
        # use inplace with care
        if top_k is not None:
            values, indices = torch.topk(similarity, k=top_k, dim=1)
            x_exp = values.exp_()
            x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
            if inplace:
                similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
                affinity = similarity
            else:
                affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
        else:
            maxes = torch.max(similarity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(similarity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum
            indices = None
        if return_usage:
            return affinity, affinity.sum(dim=2)
        return affinity

    def _readout(self, affinity, mv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mo, affinity)
        mem = mem.view(B, CV, H, W)

        return mem

    def encode_key(self, frame, need_sk=True, need_ek=True):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        f16 = self.image_encode(frame)
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])

        return key, shrinkage, selection, f16

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True):
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(masks[:, [j for j in range(num_objects) if i != j]], dim=1, keepdim=True) for i in
                range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        # frame = [1, 3, 1024, 1024], image_feat_f16 =  torch.Size([1, 256, 64, 64])
        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g16, h16

    def segment(self, single_scale_features, memory_readout,
                hidden_state, selector=None, h_out=True, strip_bg=True):

        hidden_state, logits = self.mask_decoder(single_scale_features, hidden_state, memory_readout, h_out=h_out)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'freeze_module':
            return self.freeze_module(*args, **kwargs)
        elif mode == "encode_key":
            return self.encode_key(*args, **kwargs)
        elif mode == "encode_value":
            return self.encode_value(*args, **kwargs)
        elif mode == "read_memory":
            return self.read_memory(*args, **kwargs)
        elif mode == "segment":
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def _freeze_module(self, module, except_parts=None):
        if except_parts is None:
            for name, param in module.named_parameters():
                param.requires_grad = False
        else:
            for name, param in module.named_parameters():
                param.requires_grad = False
                for p in except_parts:
                    if p in name:
                        param.requires_grad = True

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def sam_pad(self, x: torch.Tensor) -> torch.Tensor:
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def load_weights(self, state_dict, init_as_zero_if_needed=False):
        for k in list(state_dict.keys()):
            if k == 'value_encoder.network.patch_embed.proj.weight':
                if state_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    h_dim = state_dict[k].shape[0]
                    pads = torch.zeros((h_dim, 1, 16, 16), device=state_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    state_dict[k] = torch.cat([state_dict[k], pads], 1)

        self.load_state_dict(state_dict)

    def load_value_encoder_pretrained_weights(self):
        print("Loading Value Encoder Pretrained Weights")
        state_dict = torch.load(self.pretrained_weight_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'image_encoder.patch_embed.proj.weight' == k:
                h_dim = state_dict[k].shape[0]
                additional_channel = self.input_channel - 3
                pads = torch.zeros((h_dim, additional_channel, 16, 16), device=state_dict[k].device)
                nn.init.orthogonal_(pads)
                new_state_dict[k.replace("image_encoder.", "")] = torch.cat([state_dict[k], pads], dim=1)
            elif 'image_encoder' in k:
                new_state_dict[k.replace("image_encoder.", "")] = v
        self.value_encoder.network.load_state_dict(new_state_dict)
