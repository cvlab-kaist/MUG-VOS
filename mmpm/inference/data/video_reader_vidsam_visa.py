import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import torch
from typing import Any, Dict, List

import json, copy
from dataset.range_transform import im_normalization


def coco_decode_rle(encoded_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils
    h, w = encoded_rle["size"]
    encoded_rle["counts"] = encoded_rle["counts"].encode("utf-8")
    decoded_rle = mask_utils.decode(encoded_rle)
    return {"size": [h, w], "counts": decoded_rle}

def add_mask_size(masks):
    temp_masks = copy.deepcopy(masks)
    decoded_mask = [coco_decode_rle(mask) for mask in temp_masks]
    # decoded_mask = [coco_decode_rle(mask['segmentation']) for mask in temp_masks]
    for i in range(len(masks)):
        masks[i]['area'] = np.count_nonzero(decoded_mask[i]['counts'])

    return masks

class ViSAReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None, transform=None, iou_threshold=0.9, size_threshold=0):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = sorted(os.listdir(self.image_dir))
        self.palette = None# = Image.open(path.join(mask_dir, sorted(os.listdir(mask_dir))[0])).getpalette()
        self.first_gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[0])

        # if size < 0:
        #     self.im_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         im_normalization,
        #     ])
        # else:
        #     self.im_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         im_normalization,
        #         transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        #     ])
        self.size = size
        self.iou_threshold = iou_threshold
        self.transform = transform

        annotations_list = sorted([file_path for file_path in os.listdir(self.mask_dir) if
                                   file_path.endswith(".json")])
        # Threshold 떨어지는 Mask Track 제거.
        self.valid_id_set = set()
        for i, index in enumerate(annotations_list):
            with open(os.path.join(self.mask_dir, index), 'r') as f:
                json_file = json.load(f)

            for mask in json_file:
                if mask['iou'] < self.iou_threshold:
                    self.valid_id_set.add(mask['id'])

        for i, index in enumerate(annotations_list):
            if i == 0:
                with open(os.path.join(self.mask_dir, index), 'r') as f:
                    json_file = json.load(f)
                    json_file = add_mask_size(json_file)
                    self.json_file = sorted([mask for mask in json_file if mask['id'] not in list(self.valid_id_set)],
                                            key=lambda x: x['area'], reverse=True)
                self.id_order = {i: mask['id'] for i, mask in enumerate(json_file)}

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)
        info['image_dir'] = self.image_dir
    
        im_path = path.join(self.image_dir, frame)
       
        # img = Image.open(im_path).convert('RGB')

        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print(img.shape)
        # print(self.first_gt_path)
        # print(self.valid_id_set)
        # print(self.id_order)
        
        gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[idx])
        
        load_mask = self.use_all_mask or (gt_path == self.first_gt_path)
        
        masks = None
        labels = None
        if load_mask and path.exists(gt_path):
            mask_ind_id = {}
            masks = []
            labels = []
            with open(gt_path, 'r') as f:
                json_file = self.json_file
                # json_file = json.load(f)
                # json_file = add_mask_size(json_file)
                # json_file = sorted([mask for mask in json_file if mask['id'] not in list(self.valid_id_set)], key=lambda x: x['area'], reverse=True)

            for mask_ind in range(len(json_file)):
                mask_id = json_file[mask_ind]['id']
                # json_file[mask_ind] = coco_decode_rle(json_file[mask_ind]['segmentation'])
                json_file[mask_ind] = coco_decode_rle(json_file[mask_ind])
                pred_mask = json_file[mask_ind]['counts']
                mask_ind_id[mask_ind] = mask_id
                masks.append(torch.from_numpy(pred_mask).float())
                labels.append(mask_id)
                if mask_ind == 0:
                    self.palette = Image.fromarray(pred_mask).getpalette()
                #     masks = np.zeros(pred_mask.shape)
                # masks[pred_mask != 0] = mask_ind + 1
            masks = torch.stack(masks, dim=0)

        data['mask'] = masks

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        if self.transform is not None:
            img = self.transform.apply_image(img)

        img = torch.as_tensor(img)
        img = img.permute(2, 0, 1).contiguous()

        info['shape'] = shape
        info['need_resize'] = [True]
        data['rgb'] = img
        data['info'] = info
        data['labels'] = labels

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, self.size, mode='nearest')
    
    def resize_mask(self, mask, size):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, size, mode='nearest')
        
    # def resize_mask(self, mask):
    #     # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
    #     h, w = mask.shape[-2:]
    #     min_hw = min(h, w)
    #     return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
    #                 mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)