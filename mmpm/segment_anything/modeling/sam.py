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


# class Sam(nn.Module):
#     mask_threshold: float = 0.0
#     image_format: str = "RGB"
#
#     def __init__(
#             self,
#             image_encoder: ImageEncoderViT,
#             prompt_encoder: PromptEncoder,
#             mask_decoder: MaskDecoder,
#             pixel_mean: List[float] = [123.675, 116.28, 103.53],
#             pixel_std: List[float] = [58.395, 57.12, 57.375],
#     ) -> None:
#         """
#         SAM predicts object masks from an image and input prompts.
#
#         Arguments:
#           image_encoder (ImageEncoderViT): The backbone used to encode the
#             image into image embeddings that allow for efficient mask prediction.
#           prompt_encoder (PromptEncoder): Encodes various types of input prompts.
#           mask_decoder (MaskDecoder): Predicts masks from the image embeddings
#             and encoded prompts.
#           pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
#           pixel_std (list(float)): Std values for normalizing pixels in the input image.
#         """
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.prompt_encoder = prompt_encoder
#         self.mask_decoder = mask_decoder
#         self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
#
#     @property
#     def device(self) -> Any:
#         return self.pixel_mean.device
#
#     @torch.no_grad()
#     def forward(
#             self,
#             batched_input: List[Dict[str, Any]],
#             multimask_output: bool,
#     ) -> List[Dict[str, torch.Tensor]]:
#         """
#         Predicts masks end-to-end from provided images and prompts.
#         If prompts are not known in advance, using SamPredictor is
#         recommended over calling the model directly.
#
#         Arguments:
#           batched_input (list(dict)): A list over input images, each a
#             dictionary with the following keys. A prompt key can be
#             excluded if it is not present.
#               'image': The image as a torch tensor in 3xHxW format,
#                 already transformed for input to the model.
#               'original_size': (tuple(int, int)) The original size of
#                 the image before transformation, as (H, W).
#               'point_coords': (torch.Tensor) Batched point prompts for
#                 this image, with shape BxNx2. Already transformed to the
#                 input frame of the model.
#               'point_labels': (torch.Tensor) Batched labels for point prompts,
#                 with shape BxN.
#               'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
#                 Already transformed to the input frame of the model.
#               'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
#                 in the form Bx1xHxW.
#           multimask_output (bool): Whether the model should predict multiple
#             disambiguating masks, or return a single mask.
#
#         Returns:
#           (list(dict)): A list over input images, where each element is
#             as dictionary with the following keys.
#               'masks': (torch.Tensor) Batched binary mask predictions,
#                 with shape BxCxHxW, where B is the number of input prompts,
#                 C is determined by multimask_output, and (H, W) is the
#                 original size of the image.
#               'iou_predictions': (torch.Tensor) The model's predictions
#                 of mask quality, in shape BxC.
#               'low_res_logits': (torch.Tensor) Low resolution logits with
#                 shape BxCxHxW, where H=W=256. Can be passed as mask input
#                 to subsequent iterations of prediction.
#         """
#         input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
#         image_embeddings = self.image_encoder(input_images)
#
#         outputs = []
#         for image_record, curr_embedding in zip(batched_input, image_embeddings):
#             if "point_coords" in image_record:
#                 points = (image_record["point_coords"], image_record["point_labels"])
#             else:
#                 points = None
#             sparse_embeddings, dense_embeddings = self.prompt_encoder(
#                 points=points,
#                 boxes=image_record.get("boxes", None),
#                 masks=image_record.get("mask_inputs", None),
#             )
#             low_res_masks, iou_predictions = self.mask_decoder(
#                 image_embeddings=curr_embedding.unsqueeze(0),
#                 image_pe=self.prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=multimask_output,
#             )
#             masks = self.postprocess_masks(
#                 low_res_masks,
#                 input_size=image_record["image"].shape[-2:],
#                 original_size=image_record["original_size"],
#             )
#             masks = masks > self.mask_threshold
#             outputs.append(
#                 {
#                     "masks": masks,
#                     "iou_predictions": iou_predictions,
#                     "low_res_logits": low_res_masks,
#                 }
#             )
#         return outputs
#
#     def postprocess_masks(
#             self,
#             masks: torch.Tensor,
#             input_size: Tuple[int, ...],
#             original_size: Tuple[int, ...],
#     ) -> torch.Tensor:
#         """
#         Remove padding and upscale masks to the original image size.
#
#         Arguments:
#           masks (torch.Tensor): Batched masks from the mask_decoder,
#             in BxCxHxW format.
#           input_size (tuple(int, int)): The size of the image input to the
#             model, in (H, W) format. Used to remove padding.
#           original_size (tuple(int, int)): The original size of the image
#             before resizing for input to the model, in (H, W) format.
#
#         Returns:
#           (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
#             is given by original_size.
#         """
#         masks = F.interpolate(
#             masks,
#             (self.image_encoder.img_size, self.image_encoder.img_size),
#             mode="bilinear",
#             align_corners=False,
#         )
#         masks = masks[..., : input_size[0], : input_size[1]]
#         masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
#         return masks
#
#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input."""
#         # Normalize colors
#         x = (x - self.pixel_mean) / self.pixel_std
#
#         # Pad
#         h, w = x.shape[-2:]
#         padh = self.image_encoder.img_size - h
#         padw = self.image_encoder.img_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x


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
        self.value_encoder = ValueEncoderViT(value_encoder_vit, self.value_dim, self.hidden_dim, self.single_object)

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
                torch.sum(masks[:, [j for j in range(num_objects) if i != j]], dim=1, keepdim=True) for i in range(num_objects)], 1)
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
                    pads = torch.zeros((768,1,16,16), device=state_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    state_dict[k] = torch.cat([state_dict[k], pads], 1)

        self.load_state_dict(state_dict)

