import os
from os import path
import json

from inference.data.video_reader_vidsam import VideoReader
from inference.data.video_reader_vidsam_visa import ViSAReader

class LongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self, transform=None):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
                transform=transform
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        
        len_vid = len(self.vid_list)
        
        vid_split = int(len_vid / 9)
        
        spl_id = 9
        
        
        self.vid_list = self.vid_list[(spl_id-1)*vid_split:spl_id*vid_split]
        
        
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self, transform=None):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=False,
                transform=transform
            )

    def __len__(self):
        return len(self.vid_list)


class ViSATestDataset:
    def __init__(self, data_root, gt_root, size=(1080,640)):
        
        # data_list = os.listdir(data_root)
        # gt_list = os.listdir(gt_root)
        # print(os.listdir(os.path.join(data_root,data_list[0])))
        # print(os.listdir(os.path.join(gt_root,gt_list[0])))
        # print(len(data_list), len(gt_list))
        
        self.image_dir = data_root
        self.mask_dir = gt_root
        self.size = size
        self.vid_list = sorted(os.listdir(gt_root))
        
        # self.req_frame_list = {}
        # import sys;sys.exit()
        # self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        # self.mask_dir = path.join(data_root, split, 'Annotations')
        # self.size = size

        # self.vid_list = sorted(os.listdir(self.image_dir))
        # self.req_frame_list = {}

        # with open(path.join(data_root, split, 'meta.json')) as f:
        #     # read meta.json to know which frame is required for evaluation
        #     meta = json.load(f)['videos']

        #     for vid in self.vid_list:
        #         req_frames = []
        #         objects = meta[vid]['objects']
        #         for value in objects.values():
        #             req_frames.extend(value['frames'])

        #         req_frames = list(set(req_frames))
        #         self.req_frame_list[vid] = req_frames

    def get_datasets(self, transform=None):
        for video in self.vid_list:
            yield ViSAReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                transform=transform
            )

    def __len__(self):
        return len(self.vid_list)