



import os
import cv2
import numpy as np

class ViSALoader:
    def __init__(self, config):
        self.root = config['loader']['data_dir']
        
        with open('/home/cvlab11/project/kschan/a_code/videosam-data_engine/dataloaders/visa_eval_video_list.txt', 'r') as f:
            self.video_list = f.readlines()
        self.video_list = [video.strip() for video in self.video_list]
        self.num_video_clips = len(self.video_list)
        
    def __len__(self):
        return self.num_video_clips

    def get_video_clip(self, idx):
        video_name = self.video_list[idx]
        frame_paths = sorted(os.listdir(os.path.join(self.root, video_name)))
        resized_shape = (1080, 680)
        frames = []
        for frame_path in frame_paths:
            frame_path = os.path.join(self.root, video_name, frame_path)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resized_shape, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        
        return {
            'clip_name': video_name,
            'frames': frames,
        }