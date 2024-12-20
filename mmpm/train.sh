# ------------------------------------------ VIT-B ---------------------------------------------------------------
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.run --master_port 25766 \
--nproc_per_node=1 train_vidsam.py --exp_id further_train_filter_to_unfilter --stage 4 --sam_type vit_b --s4_batch_size 1 \
--s4_iterations 50000 \
--load_network /home/cvlab14/project/sangbeom/aaai/XMem/saves/pretrained_vit_b_stage_0/Aug06_23.06.21_vit_b_stage_0_s0_250000.pth