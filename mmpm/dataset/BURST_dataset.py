import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

import cv2
import json

import pycocotools.mask as cocomask

class BURST_Dataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, num_frames=3, max_num_obj=3, set_type='train', finetune=False, transform=None):
        self.im_root = os.path.join(im_root, set_type)
        self.gt_root = os.path.join(gt_root, set_type)
        self.deva_root = f'/media/dataset3/video_seg/dataset/deva/burst/{set_type}_unpreprocessed'
        
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}
        self.annos = {}
        self.image_sizes = {}

        self.deva_annos = {}

        self.transform = transform

        vid_list = sorted(os.listdir(self.im_root))
        
        if set_type == 'train':
            anno_path = os.path.join(self.gt_root, 'train.json')
            with open(anno_path, 'r') as f:    
                annotations = json.load(f)
            # https://github.com/Ali2500/BURST-benchmark/blob/main/ANNOTATION_FORMAT.md
        
        for seq in annotations['sequences']:
            seq_name = seq['seq_name']
            image_path = seq['annotated_image_paths']
            segmentations = seq['segmentations']
            masks = []
            deva_masks = []
            
            frame_path = os.path.join(self.im_root, seq_name)
            if not os.path.isdir(frame_path): 
                continue
            
            for seg in segmentations:
                frame_masks = []
                for si in seg.keys():
                    # if seg[si]['is_gt']:
                    frame_masks.append(seg[si]['rle'])
                masks.append(frame_masks)
            
            frame_path = os.path.join(self.im_root, seq_name)
            # frame_list = sorted(os.listdir(frame_path))
            frame_list = image_path
            
            frames = [os.path.join(self.im_root, frame) for frame in frame_list]            
            image_size = (seq['height'], seq['width'])

            self.frames[seq_name] = frames
            self.annos[seq_name] = masks
            self.image_sizes[seq_name] = image_size
            self.videos.append(seq_name)

            deva_path = os.path.join(self.deva_root, seq_name, 'pred.json')
            with open(deva_path, 'r') as f:    
                deva_annotations = json.load(f)
            deva_annotations = deva_annotations['annotations']
            count = 0
            for deva_anno in deva_annotations:
                if deva_anno['file_name'] == frame_list[count]:
                    count += 1
                    deva_masks.append(deva_anno['segmentations'])
            self.deva_annos[seq_name] = deva_masks 
            breakpoint()
        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        # vid_im_path = path.join(self.im_root, video)
        # vid_gt_path = path.join(self.gt_root, video)
        
        # deva_path = path.join(self.deva_root, video, 'pred.json')
        
        # with open(deva_path, 'r') as f:    
        #     deva_annotations = json.load(f)
        
        
        frames = self.frames[video]
        annos = self.annos[video]
        deva_annos = self.deva_annos[video]

        breakpoint()

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            masks_ori = []
            target_objects = []
            for f_idx in frames_idx:
                # jpg_name = frames[f_idx][:-4] + '.jpg'
                # png_name = frames[f_idx][:-4] + '.png'
                # info['frames'].append(jpg_name)

                reseed(sequence_seed)
                # this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                # this_gt = np.array(this_gt)

                # this_gt_trans = cv2.imread(path.join(vid_gt_path, png_name), 0)[:,:,None]

                this_im = cv2.imread(path.join(vid_im_path, jpg_name))
                this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)

                if self.transform is not None:
                    this_im = self.transform.apply_image(this_im)
                    this_gt_rs = cv2.resize(this_gt, this_im.shape[:2][::-1])
                    # this_gt_rs = this_gt

                this_im = torch.as_tensor(this_im)
                this_im = this_im.permute(2, 0, 1).contiguous()

                this_gt_rs = torch.as_tensor(this_gt_rs)     

                images.append(this_im)
                masks_ori.append(this_gt)
                masks.append(this_gt_rs)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)
        masks_ori = np.stack(masks_ori, 0)  

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, masks_ori.shape[-2], masks_ori.shape[-1]), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, masks.shape[-2], masks.shape[-1]), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask_ori = (masks_ori==l)
            this_mask = (masks==l)
            cls_gt[this_mask_ori] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data






















        # trials = 0
        # while trials < 5:
        info['frames'] = [] # Appended with actual frames

        num_frames = self.num_frames
        length = len(frames)
        this_max_jump = min(len(frames), self.max_jump)

        # iterative sampling
        frames_idx = [np.random.randint(length)]
        acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
        while(len(frames_idx) < num_frames):
            idx = np.random.choice(list(acceptable_set))
            frames_idx.append(idx)
            new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
            acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

        frames_idx = sorted(frames_idx)
        if np.random.rand() < 0.5:
            # Reverse time
            frames_idx = frames_idx[::-1]

        sequence_seed = np.random.randint(2147483647)
        images = []
        masks = []
        masks_ori = []
        target_objects = []

        # info['image_size'] = []
        # info['num_objects'] = []
        # info['obj_ids'] = []
        
        for f_idx in frames_idx:
            jpg_name = frames[f_idx][:-4] + '.jpg'
            png_name = frames[f_idx][:-4] + '.png'
            info['frames'].append(jpg_name)

            reseed(sequence_seed)
            this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
            this_gt = np.array(this_gt)

            # this_gt_trans = cv2.imread(path.join(vid_gt_path, png_name), 0)[:,:,None]

            this_im = cv2.imread(path.join(vid_im_path, jpg_name))
            this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                this_im = self.transform.apply_image(this_im)
                this_gt_rs = cv2.resize(this_gt, this_im.shape[:2][::-1])
                # this_gt_rs = this_gt

            this_im = torch.as_tensor(this_im)
            this_im = this_im.permute(2, 0, 1).contiguous()

            this_gt_rs = torch.as_tensor(this_gt_rs)     

            images.append(this_im)
            masks_ori.append(this_gt)
            masks.append(this_gt_rs)

            # obj_id = np.unique(this_gt)[1:]
            # info['num_objects'].append(len(obj_id))
            # info['obj_ids'].append(torch.from_numpy(obj_id))
            # info['image_size'].append(this_im.shape)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
        masks_ori = np.stack(masks_ori, 0)

        labels = np.unique(masks_ori[0])
        # Remove background
        labels = labels[labels!=0]
        target_objects = labels.tolist()

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        cls_gt = np.zeros((self.num_frames, masks_ori.shape[-2], masks_ori.shape[-1]), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, masks.shape[-2], masks.shape[-1]), dtype=np.int64)
        print(first_frame_gt.shape, target_objects)
        for i, l in enumerate(target_objects):
            this_mask_ori = (masks_ori==l)
            this_mask = (masks==1)
            cls_gt[this_mask_ori] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        print(cls_gt.max())

        # info['obj_ids'] = target_objects
        info['num_objects'] = max(1, len(target_objects))

        data = {
            'images': images,
            'first_frame_gt': first_frame_gt,
            'masks': masks,
            'masks_ori': masks_ori,
            'cls_gt': cls_gt,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)