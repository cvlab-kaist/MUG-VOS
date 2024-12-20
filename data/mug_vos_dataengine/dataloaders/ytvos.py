

import os
import cv2
import numpy as np

class YTVOSLoader:
    def __init__(self, config):
        self.root = config['loader']['data_dir']
        self.split = config['loader']['split']
        
        self.root = os.path.join(self.root, self.split, 'JPEGImages')
        self.video_list = os.listdir(self.root)
        self.video_list.sort()
        
        self.resize_shape = (1080, 680)
        self.num_video_clips = len(self.video_list)
    
    def __len__(self):
        return len(self.video_list)
    
    def get_video_clip(self, idx):
        if isinstance(idx, str):
            video_name = idx
        else:
            video_name = self.video_list[idx]
        frame_paths = sorted(os.listdir(os.path.join(self.root, video_name)))
        
        frames = []
        for frame_path in frame_paths:
            frame_path = os.path.join(self.root, video_name, frame_path)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        
        return {
            'clip_name': video_name,
            'frames': frames,
            'annos': None,
        }