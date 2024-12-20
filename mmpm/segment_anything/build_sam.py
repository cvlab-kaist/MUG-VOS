# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, VidSam, TwoWayTransformer, Transformer


def build_sam_vit_h(checkpoint=None, single_object=False):
    print("Using SAM Huge")
    vit_path = "/media/data3/video_seg/trained_model/sam/sam_vit_h_4b8939.pth"

    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        single_object=single_object,
        base_path=vit_path
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, single_object=False):
    print("Using SAM Large")
    vit_path = "/media/data3/video_seg/trained_model/sam/sam_vit_l_0b3195.pth"

    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        single_object=single_object,
        base_path=vit_path
    )


def build_sam_vit_b(checkpoint=None, single_object=False):
    print("Using SAM Base")
    vit_path = "/media/data3/video_seg/trained_model/sam/sam_vit_b_01ec64.pth"

    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        single_object=single_object,
        checkpoint=checkpoint,
        base_path=vit_path
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        single_object=False,
        checkpoint=None,
        base_path = None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    transform_embed_dim = 512

    key_dim = 64
    value_dim = 512
    hidden_dim = 64

    sam = VidSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        mask_decoder=MaskDecoder(
            value_dim, hidden_dim, transformer_dim=transform_embed_dim
        ),
        hidden_dim=hidden_dim, key_dim=key_dim, value_dim=value_dim,
        single_object=single_object,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    # sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        init_as_zero_if_needed = True
        import torch.nn as nn

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

        sam.load_state_dict(state_dict)
        print("Successfully loaded pretrained weights")
    else:
        print("Loading Query Encoder Pretrained Weights")
        with open(base_path, "rb") as f:  # VIT - H
            state_dict = torch.load(f)

        new_state_dict = {}
        for k, v in state_dict.items():
            if 'image_encoder' in k:
                new_state_dict[k.replace("image_encoder.","")] = v
        sam.image_encoder.load_state_dict(new_state_dict)

    return sam
