# ------------------------------------- MUG-TEST Full Version -------------------------------------

# Filtered
mem_interval=3
CUDA_VISIBLE_DEVICES='6' python -m torch.distributed.run --master_port 25715 --nproc_per_node=1 eval.py \
--exp_id evaluation --stage 1 --load_network /home/cvlab14/project/sangbeom/aaai/XMem/saves/Aug10_01.08.20_stage_4_vit_b_s4/Aug10_01.08.20_stage_4_vit_b_s4_270000.pth \
--mem_interval $mem_interval --threshold 0.5 \
--sam_type vit_b --output_path /home/cvlab14/project/sangbeom/MUG/results/full_version/main/mmpm/filtered/270k/$threshold/mem_int_$mem_interval