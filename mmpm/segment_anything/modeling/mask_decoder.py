# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d

from model.modules import FeatureFusionBlock, GConv2D


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims, mid_dim, kernel_size=1)
        self.g_dims = g_dims

        self.transform = GConv2D(mid_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g)

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, :, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, :, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h


class MaskDecoder(nn.Module):
    def __init__(
            self, val_dim, hidden_dim, transformer_dim: int, activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.fuser = FeatureFusionBlock(256, val_dim + hidden_dim, 512, 512)  # 1024 -> 256 으로 바뀜
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(512 + 1, 256, hidden_dim)
        else:
            self.hidden_update = None

        self.pred = nn.Conv2d(512, 1, kernel_size=3, padding=1, stride=1)  # 원래 256 -> 64 로 바꿈

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        #     activation(),
        # )

    def forward(
            self, f16, hidden_state, memory_readout, h_out=True) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        # b, t, c, h, w = g16.shape  # [b,t, 512, 24, 24]
        # upscaled_embedding = self.output_upscaling(g16.view(b * t, c, h, w))  # [1, 64, 256, 256]
        # upscaled_embedding = upscaled_embedding.view(b, t, *upscaled_embedding.shape[1:])
        # 원래는 [16, 1, 256, 96, 96] = upscaled_embedding 이랬음.

        logits = self.pred(F.relu(g16.flatten(start_dim=0, end_dim=1)))  # [b,t,64,64]

        if h_out and self.hidden_update is not None:
            g16 = torch.cat([g16, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2) # c가 +1 되어 있음.
            # g16 = [b,t, 512 +1, 64, 64], hidden_state = [b,t, 64,64,64]
            hidden_state = self.hidden_update(g16, hidden_state)
        else:
            hidden_state = None

        logits = F.interpolate(logits, scale_factor=16, mode='bilinear', align_corners=False) # [1,1,64,64] -> [1,1,256,256]
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0] // 3
            print(f'Hyperparameters read from the model weights: '
                  f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights
