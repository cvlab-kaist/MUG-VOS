from argparse import ArgumentParser


def none_or_default(x, default):
    return x if x is not None else default

class Configuration():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        parser.add_argument('--no_amp', action='store_true')

        # Data parameters
        parser.add_argument('--static_root', help='Static training data root', default='/media/data3/sj/static')
        parser.add_argument('--bl_root', help='Blender training data root', default='../BL30K')
        parser.add_argument('--yv_root', help='YouTubeVOS data root', default='/media/data3/sj/YouTube')
        parser.add_argument('--davis_root', help='DAVIS data root', default='/media/data3/sj/DAVIS')
        parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=16)

        parser.add_argument('--key_dim', default=64, type=int)
        parser.add_argument('--value_dim', default=512, type=int)
        parser.add_argument('--hidden_dim', default=64, help='Set to =0 to disable', type=int)

        parser.add_argument('--deep_update_prob', default=0.2, type=float)

        parser.add_argument('--stages', help='Training stage (0-static images, 1-Blender dataset, 2-DAVIS+YouTubeVOS)', default='02')

        """
        Stage-specific learning parameters
        Batch sizes are effective -- you don't have to scale them when you scale the number processes
        """
        # Stage 0, DAVIS+YoutubeVOS, Evaluation
        parser.add_argument('--s0_batch_size', default=1, type=int)
        parser.add_argument('--s0_iterations', default=150000, type=int)
        # fine-tune means fewer augmentations to train the sensory memory
        parser.add_argument('--s0_finetune', default=10000, type=int)
        parser.add_argument('--s0_steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--s0_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s0_num_ref_frames', default=1, type=int)
        parser.add_argument('--s0_num_frames', default=-1, type=int)
        parser.add_argument('--s0_start_warm', default=20000, type=int)
        parser.add_argument('--s0_end_warm', default=70000, type=int)

        # Stage 1, ViSA Evaluation, Evaluation
        parser.add_argument('--s1_batch_size', default=1, type=int)
        parser.add_argument('--s1_iterations', default=150000, type=int)
        # fine-tune means fewer augmentations to train the sensory memory
        parser.add_argument('--s1_finetune', default=10000, type=int)
        parser.add_argument('--s1_steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--s1_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s1_num_ref_frames', default=1, type=int)
        parser.add_argument('--s1_num_frames', default=-1, type=int)
        parser.add_argument('--s1_start_warm', default=20000, type=int)
        parser.add_argument('--s1_end_warm', default=70000, type=int)

        parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.05, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

        # Evaluation Functions
        parser.add_argument('--max_memory_length', help='Maximum length of Memory, decrease to save memory', type=int,
                            default=10)
        parser.add_argument('--mem_interval', help='Memory Interval.', type=int, default=3)
        parser.add_argument('--memory_filtering_method', help='Memory filtering method when memory rich predefined length',
                            type=str, default="random_except_first_last")
        parser.add_argument('--top_k', type=int, default=30)
        parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int,
                            default=-1)

        # Ablation study 4
        parser.add_argument("--no_memory", help="No memory", action="store_true")
        parser.add_argument("--add_sequential_memory", help="add_sequential_memory", action="store_true")
        parser.add_argument("--add_temporal_memory", help="add_temporal_memory", action="store_true")
        parser.add_argument("--only_temporal_memory", help="Only temporal Memory", action="store_true")

        # Logging information
        parser.add_argument('--log_text_interval', default=100, type=int)
        parser.add_argument('--log_image_interval', default=1000, type=int)
        parser.add_argument('--save_network_interval', default=10000, type=int)
        parser.add_argument('--save_checkpoint_interval', default=50000, type=int)
        parser.add_argument('--exp_id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        # Video SAM
        parser.add_argument('--sam_type', default='default', type=str, help="default, vit_h, vit_l, vit_b")

        # Evaluation
        parser.add_argument("--output_path", default="./output", type=str, help="Output path")
        parser.add_argument("--threshold", default=0.5, type=float, help="Mask Threshold")
        # parser.add_argument(
        #     '--threshold',
        #     nargs='+',  # This allows for a list of arguments
        #     type=float,  # Ensures each argument is a float
        #     default=[0.1, 0.2, 0.3, 0.4, 0.5]
        #     # Provide a default list of floats
        # )

        # # Multiprocessing parameters, not set by users
        # parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        self.args['amp'] = not self.args['no_amp']

        # check if the stages are valid
        stage_to_perform = list(self.args['stages'])
        for s in stage_to_perform:
            if s not in ['0', '1', '2', '3','4','eval']:
                raise NotImplementedError

    def get_stage_parameters(self, stage):
        parameters = {
            'batch_size': self.args['s%s_batch_size'%stage],
            'iterations': self.args['s%s_iterations'%stage],
            'finetune': self.args['s%s_finetune'%stage],
            'steps': self.args['s%s_steps'%stage],
            'lr': self.args['s%s_lr'%stage],
            'num_ref_frames': self.args['s%s_num_ref_frames'%stage],
            'num_frames': self.args['s%s_num_frames'%stage],
            'start_warm': self.args['s%s_start_warm'%stage],
            'end_warm': self.args['s%s_end_warm'%stage],
        }

        return parameters

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
