'''
    This code is used to generate data from video clips.
'''

import yaml
import argparse
import torch
# custom imports
from dataloaders import get_loader
from data_engine import get_engine

def run(config):
    dataset = config['loader']['dataset']
    print('init loader')
    loader = get_loader(config)
    print('init engine')
    engine = get_engine(config)

    assert dataset == 'DAVIS', 'Only DAVIS dataset is supported'

    if dataset == 'DAVIS' and config['gen_method']['use_bin']:
        print(f'Dataset: {dataset} with use_bin {config["gen_method"]["use_bin"]}')
        for video_clip in range(loader.num_video_clips):
            video_clip = loader.get_video_clip(video_clip)
            print(f"Processing video clip {video_clip['clip_name']}")

            bin_size = config['gen_method']['bin_size']
            n_frames = len(video_clip['frames'])
            n_bins = n_frames // bin_size

            for bin_idx in range(n_bins):
                bin_start = bin_idx * bin_size
                bin_end = bin_start + bin_size
                video_clip_bin = {
                    'clip_name': video_clip['clip_name'] + f'_bin{bin_idx}',
                    'frames': video_clip['frames'][bin_start:bin_end],
                    'annos': video_clip['annos'],
                }
                print('Processing video clip bin', video_clip_bin['clip_name'])
                engine.process_video_clip(video_clip_bin)
    elif dataset == 'DAVIS' and config['gen_method']['use_bin'] == False:
        print(f'Dataset: {dataset} with use_bin {config["gen_method"]["use_bin"]}')
        for video_clip in range(loader.num_video_clips):
            video_clip = loader.get_video_clip(video_clip)
            print(f"Processing video clip {video_clip['clip_name']}")
            # if video_clip['clip_name'] in ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog']:
            #     continue
            engine.process_video_clip(video_clip)
    print('Data generation completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--config', default='configs/engine-v1-davis-val-2.yaml', type=str, help='Path to config file')
    args = parser.parse_args()
    # load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # set device
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # run data generation
    run(config=config)