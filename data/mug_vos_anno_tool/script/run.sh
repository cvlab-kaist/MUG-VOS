# 이 코드는 visa dataset을 생성하기 위한 annotation tool을 실행하는 코드입니다.
CUDA_VISIBLE_DEVICES=0 python app.py \
    --port 3001 \
    --video_dir /home/cvlab14/project/seongchan/mug/mug_vos_anno_tool/test_videos \
    --anno_dir /home/cvlab14/project/seongchan/mug/mug_vos_anno_tool/test_masks \
    --vis_dir /home/cvlab14/project/seongchan/mug/mug_vos_anno_tool/test_vis