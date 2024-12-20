import datetime
from os import path
import math
import git

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.trainer import XMemTrainer
from dataset.static_dataset_sam import StaticTransformDataset
# from dataset.static_dataset import StaticTransformDataset

# from dataset.vos_dataset import VOSDataset
from dataset.vid_sam_dataset import VID_SAM_Dataset, VID_SAM_Dataset_eval
from dataset.BURST_dataset import BURST_Dataset
from dataset.ViSA_dataset import ViSA_Eval_Dataset

from util.logger import TensorboardLogger
# from util.configuration import Configuration
from util.eval_configuration import Configuration
from util.load_subset import load_sub_davis, load_sub_yv

from segment_anything import SamTrainer, Eval_SamTrainer
from segment_anything import ResizeLongestSide

from tqdm import tqdm

"""
Initial setup
"""
# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
raw_config = Configuration()
raw_config.parse()

if raw_config['benchmark']:
    torch.backends.cudnn.benchmark = True

# Get current git info
repo = git.Repo("../")
git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')

network_in_memory = None
stages = raw_config['stages']
stages_to_perform = list(stages)

for si, stage in enumerate(stages_to_perform):

    # Set seed to ensure the same initialization
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    # Pick stage specific hyperparameters out
    stage_config = raw_config.get_stage_parameters(stage)
    config = dict(**raw_config.args, **stage_config)
    if config['exp_id'] != 'NULL':
        config['exp_id'] = config['exp_id'] + '_s%s' % stages[:si + 1]

    config['single_object'] = False

    config['num_gpus'] = world_size
    if config['batch_size'] // config['num_gpus'] * config['num_gpus'] != config['batch_size']:
        raise ValueError('Batch size must be divisible by the number of GPUs.')
    config['batch_size'] //= config['num_gpus']
    config['num_workers'] //= config['num_gpus']
    print(f'We are assuming {config["num_gpus"]} GPUs.')

    print(f'We are now starting stage {stage}')

    """
    Model related
    """
    if local_rank == 0:
        # import wandb
        #
        # wandb.init(project="my-project", sync_tensorboard=True)

        # Logging
        if config['exp_id'].lower() != 'null':
            print('I will take the role of logging!')
            long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
        else:
            long_id = None
        logger = TensorboardLogger(config['exp_id'], long_id, git_info)
        logger.log_string('hyperpara', str(config))

        # Construct the rank 0 model
        model = Eval_SamTrainer(config, logger=logger,
                           save_path=path.join('saves', long_id, long_id) if long_id is not None else None,
                           local_rank=local_rank, world_size=world_size).test()
    else:
        # Construct model for other ranks
        model = Eval_SamTrainer(config, local_rank=local_rank, world_size=world_size).test()

    # Load pertrained model if needed
    if raw_config['load_checkpoint'] is not None:
        total_iter = model.load_checkpoint(raw_config['load_checkpoint'])
        raw_config['load_checkpoint'] = None
        print('Previously trained model loaded!')
    else:
        total_iter = 0

    # if raw_config['load_network'] is not None:
    #     # from segment_anything.build_sam import (
    #     #     build_sam,
    #     #     build_sam_vit_h,
    #     #     build_sam_vit_l,
    #     #     build_sam_vit_b,
    #     #     sam_model_registry,
    #     # )
    #     #
    #     model.load_network(raw_config['load_network'])
    #     # network = sam_model_registry[config['sam_type']](checkpoint=raw_config['load_network'])
    #     raw_config['load_network'] = None

    # if network_in_memory is not None:
    #     print('I am loading network from the previous stage')
    #     model.load_network_in_memory(network_in_memory)
    #     network_in_memory = None
    # elif raw_config['load_network'] is not None:
    #     print('I am loading network from a disk, as listed in configuration')
    #     model.load_network(raw_config['load_network'])
    #     raw_config['load_network'] = None

    """
    Dataloader related
    """


    # To re-seed the randomness everytime we start a worker
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 31) + worker_id + local_rank * 100
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def construct_loader(dataset):
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
        train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler,
                                  num_workers=config['num_workers'],
                                  worker_init_fn=worker_init_fn, drop_last=True)
        return train_sampler, train_loader


    def renew_vos_loader(max_skip, finetune=False, transform=None):
        # //5 because we only have annotation for every five frames
        # yv_dataset = VID_SAM_Dataset(path.join(yv_root, 'JPEGImages'),
        #                              path.join(yv_root, 'Annotations'), max_skip // 5, is_bl=False,
        #                              subset=load_sub_yv(), num_frames=config['num_frames'], finetune=finetune,
        #                              transform=transform)

        davis_dataset = VID_SAM_Dataset_eval(path.join(davis_root, 'JPEGImages', '480p'),
                                        path.join(davis_root, 'Annotations', '480p'), max_skip,
                                        is_bl=False, subset=load_sub_davis(), num_frames=config['num_frames'],
                                        eval=True, transform=transform)

        # train_dataset = ConcatDataset([davis_dataset]*5 + [yv_dataset])

        # print(f'YouTube dataset size: {len(yv_dataset)}')
        # print(f'DAVIS dataset size: {len(davis_dataset)}')
        # print(f'Concat dataset size: {len(train_dataset)}')
        # print(f'Renewed with {max_skip=}')

        # return construct_loader(yv_dataset)
        return construct_loader(davis_dataset)

    def renew_visa_loader(max_skip, finetune=False, transform=None):
        import os
        visa_path = "/media/data3/ref_vseg/datasets/DAVIS/JPEGImages/480p"

        visa_anno_path = "/media/data3/video_seg/dataset/mug_vos_sb/mug_vos_test" # Previous ViSA
        # visa_anno_path = "/media/data3/sangbeom/temp/mug_vos" # Previous ViSA
        train_dataset = ViSA_Eval_Dataset(visa_path, visa_anno_path, max_skip, is_bl=True, num_frames=config['num_frames'],
                                     eval=True, transform=transform)

        return construct_loader(train_dataset)


    """
    Dataset related
    """

    """
    These define the training schedule of the distance between frames
    We will switch to max_skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
    Not effective for stage 0 training
    The initial value is not listed here but in renew_vos_loader(X)
    """
    max_skip_values = [10, 15, 5, 5]
    transform = ResizeLongestSide(model.vid_sam.image_encoder.img_size)

    # renew_visa_loader(5)
    # breakpoint()

    if stage == '0':
        # DAVIS
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        # VOS dataset, 480p is used for both datasets
        yv_root = path.join(path.expanduser(config['yv_root']), 'train_480p')
        davis_root = path.join(path.expanduser(config['davis_root']), '2017', 'trainval')

        train_sampler, train_loader = renew_vos_loader(1, transform=transform)
        # train_sampler, train_loader = renew_visa_loader(5, transform=transform)
        renew_loader = renew_vos_loader
        
    elif stage == '1': # ViSA EVAL
        train_sampler, train_loader = renew_visa_loader(1, transform=transform)
        renew_loader = renew_vos_loader


    elif stage == '4':
        # stage 2 or 3
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        # VOS dataset, 480p is used for both datasets
        yv_root = path.join(path.expanduser(config['yv_root']), 'train_480p')
        davis_root = path.join(path.expanduser(config['davis_root']), '2017', 'trainval')

        train_sampler, train_loader = renew_vos_loader(1, transform=transform)
        # train_sampler, train_loader = renew_visa_loader(5, transform=transform)
        renew_loader = renew_vos_loader

    # burst_dataset = renew_burst_loader(5, transform=transform)
    # burst_sampler, burst_loader = construct_loader(burst_dataset)

    # for data in burst_dataset:
    #     pass
    # breakpoint()

    """
    Determine max epoch
    """

    assert config['batch_size'] == 1
    iteration = len(train_loader)
    total_epoch = 1

    print(f'We approximately use {total_epoch} epochs.')
    current_epoch = 0

    """
    Starts training
    """
    finetuning = False
    # Need this to select random bases in different workers
    np.random.seed(np.random.randint(2 ** 30 - 1) + local_rank * 100)
    try:
        while total_iter < iteration:

            # Crucial for randomness!
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            # Train loop
            model.test()
            for data in tqdm(train_loader):
                model.do_eval(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break
    finally:
        if not config['debug'] and model.logger is not None and total_iter > 5000:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)

distributed.destroy_process_group()
