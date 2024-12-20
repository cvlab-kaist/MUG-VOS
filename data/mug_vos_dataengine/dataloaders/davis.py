import os
import cv2
import json
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torchvision.transforms.functional as torchvision_f

class DAVISLoader:
    def __init__(self, config):
        self.root = config['loader']['data_dir']
        self.split = config['loader']['split']
        # load video list from split file
        with open(f'{self.root}/ImageSets/2017/{self.split}.txt', 'r') as f:
            self.video_list = f.read().split('\n')
        self.video_list.sort()
        self.num_video_clips = len(self.video_list)

    def __len__(self):
        return self.num_video_clips

    def get_video_clip(self, idx):
        '''
        args:
            idx: int, index of the video clip
        returns:
            frames: list of np.ndarray, list of frames in the video clip
            annos: list of dict, list of annotations at the first frame in the video clip
        '''
        if type(idx) == str:
            video_name = idx
        elif type(idx) == int:
            video_name = self.video_list[idx]
        # img_resolutions = 'JPEGImages/Full-Resolution'
        img_resolutions = 'JPEGImages/480p'
        frame_paths = sorted(os.listdir(os.path.join(self.root, img_resolutions, video_name)))
        if img_resolutions == 'JPEGImages/Full-Resolution':
            resized_shape = (1920, 1080)
        elif img_resolutions == 'JPEGImages/480p':
            resized_shape = (856, 480)
        frames = []
        for frame_path in frame_paths:
            frame_path = os.path.join(self.root, img_resolutions, video_name, frame_path)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resized_shape, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        
        with open(os.path.join(self.root, 'RLE_Annotations', video_name, '00000_final.json'), 'r') as f:
            annotations = json.load(f)

        annos = []
        for encoded_anno in annotations:
            decoded_anno = self.decode_rle(encoded_anno)
            decoded_anno['counts'] = cv2.resize(decoded_anno['counts'], resized_shape, interpolation=cv2.INTER_LINEAR)
            decoded_anno['counts'] = np.round(decoded_anno['counts']) > 0
            annos.append(decoded_anno)

        return {
            "clip_name": video_name,
            "frames": frames, 
            "annos": annos,
        }
    
    def decode_rle(self, encoded_rle):
        '''
        args:
            encoded_rle: dict, encoded rle annotation
        returns:
            decoded_rle: dict, decoded rle annotation
        '''
        encoded_rle["counts"] = encoded_rle["counts"].encode("utf-8")
        return {
            "size": encoded_rle["size"], 
            "counts": mask_utils.decode(encoded_rle),
            "id": encoded_rle["id"]
        }