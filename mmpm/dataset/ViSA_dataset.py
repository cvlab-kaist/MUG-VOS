import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

import copy

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

import cv2

import json

import pycocotools.mask as cocomask

class ViSA_Eval_Dataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, num_frames=-1, max_num_obj=3, eval=True, transform=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}
        self.annos = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            num_frames = len(os.listdir(os.path.join(self.im_root, vid)))
            if os.path.exists(os.path.join(self.gt_root, vid)):
                annos = sorted(os.listdir(os.path.join(self.gt_root, vid)))
            else:
                continue


            for vid_anno in annos:
                vid_vidanno = os.path.join(vid, vid_anno)
                self.frames[vid_vidanno] = sorted(os.listdir(os.path.join(self.im_root, vid)))

                # self.annos[vid] = sorted(os.listdir(os.path.join(self.gt_root, vid, vid_anno)))
                self.videos.append(vid_vidanno)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))


        self.transform = transform

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {} 
        video_name_track = copy.deepcopy(video)
        video_name = copy.deepcopy(video).split("/")[0]
        info['name'] = video_name_track.replace("/", "_")

        vid_im_path = path.join(self.im_root, video_name)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]
        # annos = self.annos[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            if self.num_frames == -1:
                num_frames = len(frames)
            else:
                num_frames = self.num_frames

            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # Sorted Sampling
            frames_idx = [i for i in range(length)]


            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            masks_ori = []
            target_objects = []

            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][2:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = np.array(this_gt)

                this_im = cv2.imread(path.join(vid_im_path, jpg_name))
                this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)
                origin_size = this_im.shape[:2]

                if self.transform is not None:
                    this_im = self.transform.apply_image(this_im)
                    this_gt_rs = cv2.resize(this_gt, this_im.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

                this_im = torch.as_tensor(this_im)
                this_im = this_im.permute(2, 0, 1).contiguous()

                this_gt_rs = torch.as_tensor(this_gt_rs)

                images.append(this_im)
                masks_ori.append(this_gt)
                masks.append(this_gt_rs)

            images = torch.stack(images, dim=0)

            labels = np.unique(masks[0])
            # Remove Background
            labels = labels[labels != 0]

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()

            if len(target_objects) > self.max_num_obj:
                target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

            info['num_objects'] = max(1, len(target_objects))
            max_num_obj = len(target_objects)

            masks = np.stack(masks, 0)
            masks_ori = np.stack(masks_ori, 0)

            # Generate one-hot ground-truth
            cls_gt = np.zeros((num_frames, masks.shape[-2], masks.shape[-1]), dtype=np.int64)
            first_frame_gt = np.zeros((1, max_num_obj, masks.shape[-2], masks.shape[-1]), dtype=np.int64)
            for i, l in enumerate(target_objects):
                this_mask_ori = (masks_ori == l)
                this_mask = (masks == l)
                # cls_gt[this_mask_ori] = i+1
                cls_gt[this_mask] = i + 1
                first_frame_gt[0, i] = (this_mask[0])
            cls_gt = np.expand_dims(cls_gt, 1)

            # 1 if object exist, 0 otherwise
            selector = [1 if i < info['num_objects'] else 0 for i in range(max_num_obj)]
            selector = torch.FloatTensor(selector)

            data = {
                'rgb': images,
                'first_frame_gt': first_frame_gt,
                'cls_gt': cls_gt,
                'selector': selector,
                'info': info,
                'pallete': False,
                'origin_size': origin_size,
            }

            return data

    def __len__(self):
        return len(self.videos)