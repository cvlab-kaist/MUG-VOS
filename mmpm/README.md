# PyTorch Implementation of MMPM

## Preparing the Environment

```bash

conda create -n mmpm python=3.8
conda activate mmpm

pip -r requirements.txt
```

## MMPM Train & Evaluation

### 1. Static Image train
```bash
python -m torch.distributed.run --master_port 25769 --nproc_per_node={GPU_NUMs} train_vidsam.py \
--exp_id vit_b_stage_0 --stage 0 --sam_type vit_b --s0_batch_size # Your Batch Size
```

### 2. MUG-VOS train
```bash
python -m torch.distributed.run --master_port 25766 \
--nproc_per_node={GPU_NUMs} train_mmpm.py --exp_id stage_4 --stage 4 --sam_type vit_b --s4_batch_size {Batch_Size} \
--load_network {Stage 0 Checkpoint}
```

In `util/configuration.py`, can change more hyperparameters.

### 3. Run Evaluation

To evaluate the mmpm model
```bash
mem_interval=3
python -m torch.distributed.run --master_port 25715 --nproc_per_node={GPU_NUMs} eval.py \
--exp_id evaluation --stage 1 --load_network {Model Checkpoint} \
--mem_interval $mem_interval --threshold 0.5 \
--sam_type vit_b --output_path {Save Path}
```