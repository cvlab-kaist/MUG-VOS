import cv2
import numpy as np
from os import path
import os
from PIL import Image

import torch
from dataset.range_transform import inv_im_trans
from collections import defaultdict

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def base_transform(im, size):
    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    # Resize
    if im.shape[1] != size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

    return im

def im_transform(im, size):
    return base_transform(detach_to_cpu(im / 255.), size=size)

def mask_transform(mask, size):
    return base_transform(mask, size=size)


def out_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)

def merge_masks(masks):
    non_zero_mask = torch.sum(masks, dim=0).detach().cpu().numpy()

    masks = torch.max(masks, dim=0).indices
    max_object_num = torch.max(masks).int()

    masks = (masks.detach().cpu().numpy()).astype(np.uint8)

    new_mask = np.zeros_like(masks)

    for i in range(0, max_object_num+1):
        new_mask[masks == i] = i + 1

    new_mask[non_zero_mask == 0] = 0

    return new_mask


def save_results(images, size, output_path, threshold=0.5, pallete=True):
    b, t = images['rgb'].shape[:2]
    h, w = size

    os.makedirs(output_path, exist_ok=True)

    for bi in range(b):
        for ti in range(t):
            # image = (im_transform(images['rgb'][bi,ti], size))

            if pallete[0] == False:
                if ti == 0:
                    mask = mask_transform(merge_masks(images['first_frame_gt'][bi][0]), (w, h))
                else:
                    mask = mask_transform(merge_masks(images['masks_%d' % ti][bi] > threshold), (w, h))

                mask = mask * 255
                out_img = Image.fromarray(mask)

            else:
                # For Davis
                img_pallete_source = Image.open(
                    '/media/data3/sj/DAVIS/2017/trainval/Annotations/480p/bear/00000.png').convert('P')
                img_pallete = img_pallete_source.getpalette()

                if ti == 0:
                    mask = mask_transform(merge_masks(images['first_frame_gt'][bi][0]), (w, h))
                else:
                    merge_mask = np.zeros((h, w))
                    for i, mask in enumerate(images['masks_%d' % ti][bi]):
                        mask = mask_transform((mask > 0.5).detach().cpu().numpy().astype(np.uint8), (w, h))
                        merge_mask[mask == 1] = i + 1

                    mask = merge_mask.astype(np.uint8)

                out_img = Image.fromarray(mask)
                out_img.putpalette(img_pallete)

                out_img.save(path.join(output_path, str(ti).zfill(5) + '.png'))


            out_img.save(path.join(output_path, str(ti).zfill(5) + '.png'))

# def mask_indexer(mask_size):
#     new_mask = np.zeros_like(mask_size)
#     for l, i in self.remappings.items():
#         new_mask[mask == i] = l
#     return new_mask
#
# # Save the mask
# def save_mask_track(out_path, vid_name):
#     this_out_path = path.join(out_path, vid_name)
#     os.makedirs(this_out_path, exist_ok=True)
#     out_mask = mapper.remap_index_mask(out_mask)
#     out_img = Image.fromarray(out_mask)
#     if vid_reader.get_palette() is not None:
#         out_img.putpalette(vid_reader.get_palette())
#     out_img.save(os.path.join(this_out_path, frame[:-4] + '.png'))
#
# if args.save_scores:
#     np_path = path.join(args.output, 'Scores', vid_name)
#     os.makedirs(np_path, exist_ok=True)
#     if ti == len(loader) - 1:
#         hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
#     if args.save_all or info['save'][0]:
#         hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')