import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

import cv2


class VID_SAM_Dataset_eval(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, num_frames=-1, max_num_obj=100, eval=True,
                 transform=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}

        self.transform = transform

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None and not eval:
                if vid not in subset:
                    continue
            elif subset is not None and eval:
                if vid in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))

            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0

        info['frames'] = []  # Appended with actual frames

        if self.num_frames == -1:
            num_frames = len(frames)
        else:
            num_frames = self.num_frames

        length = len(frames)
        this_max_jump = min(len(frames), self.max_jump)

        # Sorted
        frames_idx = [i for i in range(length)]

        # # iterative sampling
        # frames_idx = [np.random.randint(length)]
        # acceptable_set = set(
        #     range(max(0, frames_idx[-1] - this_max_jump), min(length, frames_idx[-1] + this_max_jump + 1))).difference(
        #     set(frames_idx))
        # while (len(frames_idx) < num_frames):
        #     idx = np.random.choice(list(acceptable_set))
        #     frames_idx.append(idx)
        #     new_set = set(
        #         range(max(0, frames_idx[-1] - this_max_jump), min(length, frames_idx[-1] + this_max_jump + 1)))
        #     acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

        sequence_seed = np.random.randint(2147483647)
        images = []
        masks = []
        masks_ori = []
        target_objects = []

        for f_idx in frames_idx:
            jpg_name = frames[f_idx][:-4] + '.jpg'
            png_name = frames[f_idx][:-4] + '.png'
            info['frames'].append(jpg_name)

            reseed(sequence_seed)
            this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
            pallete = this_gt.getpalette()

            this_gt = np.array(this_gt)

            # this_gt_trans = cv2.imread(path.join(vid_gt_path, png_name), 0)[:,:,None]

            this_im = cv2.imread(path.join(vid_im_path, jpg_name))
            this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)
            origin_size = this_im.shape[:2]

            if self.transform is not None:
                this_im = self.transform.apply_image(this_im)
                this_gt_rs = cv2.resize(this_gt, this_im.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                # this_gt_rs = this_gt

                # # Debug
                # pallete = this_gt.getpalette()
                # temp_gt = Image.fromarray(this_gt_rs)
                # temp_gt.putpalette(pallete)
                # temp_gt.save("/home/cvlab14/project/sangbeom/aaai/gt.png")

            this_im = torch.as_tensor(this_im)
            this_im = this_im.permute(2, 0, 1).contiguous()

            this_gt_rs = torch.as_tensor(this_gt_rs)

            images.append(this_im)
            masks_ori.append(this_gt)
            masks.append(this_gt_rs)

        images = torch.stack(images, 0)

        labels = np.unique(masks[0])
        # Remove background
        labels = labels[labels != 0]

        if len(labels) == 0:
            target_objects = []
            trials += 1
        else:
            target_objects = labels.tolist()

        # ----------------------------------------------------------------- #
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
            'pallete': pallete,
            'origin_size': origin_size,
        }

        return data

    def __len__(self):
        return len(self.videos)




class VID_SAM_Dataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, num_frames=3, max_num_obj=1, finetune=False, transform=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15,
                                    shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.BILINEAR,
                                    fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15,
                                    shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.NEAREST,
                                    fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36, 1.00), interpolation=InterpolationMode.BILINEAR)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36, 1.00), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            # im_normalization,
        ])

        self.transform = transform

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

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
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')

                pallete = this_gt.getpalette()

                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                if self.transform is not None:
                    this_im = this_im.permute(1, 2, 0).numpy()
                    this_im = self.transform.apply_image(np.array(this_im))

                    this_gt_rs = cv2.resize(this_gt, this_im.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

                    # # Debug
                    # temp_gt = Image.fromarray(this_gt_rs)
                    # temp_gt.putpalette(pallete)
                    # temp_gt.save(f"/home/cvlab14/project/sangbeom/aaai/gt_{f_idx}.png")

                this_im = torch.as_tensor(this_im)
                this_im = this_im.permute(2, 0, 1).contiguous()

                this_gt = np.array(this_gt)
                # this_gt_rs = torch.as_tensor(this_gt_rs)

                images.append(this_im)
                masks_ori.append(this_gt)
                masks.append(this_gt_rs)

            images = torch.stack(images, 0)
            
            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]
            
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
        # cls_gt = np.zeros((self.num_frames, masks_ori.shape[-2], masks_ori.shape[-1]), dtype=np.int64)
        cls_gt = np.zeros((self.num_frames, masks.shape[-2], masks.shape[-1]), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, masks.shape[-2], masks.shape[-1]), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask_ori = (masks_ori==l)
            this_mask = (masks==l)
            # cls_gt[this_mask_ori] = i+1
            cls_gt[this_mask] = i+1
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


    def __len__(self):
        return len(self.videos)