CUDA_VISIBLE_DEVICES=0 python app_add.py \
    --port 3001 \
    --video_dir /home/cvlab14/project/seongchan/mug/mug_vos_anno_tool/test_videos \
    --anno_dir /home/cvlab14/project/seongchan/mug/mug_vos_anno_tool/test_masks \
    --vis_dir /home/cvlab14/project/seongchan/mug/mug_vos_anno_tool/test_vis
# 이 코드는 첫번째 프레임을 기준으로 이미 생성되어 있는 마스크들과 겹치지 않는 새로운 마스크를 추가하기 위한 코드입니다.