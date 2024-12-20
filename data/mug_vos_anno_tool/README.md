# MUG-VOS DATA ANNOTATION TOOLS

This project is to utilize Segment Anything Model (SAM) to annotate video data.
As well as Segment Anything Model (SAM), we also use RAFT, optical flow model, to track objects in the video.

## Installation
First, you need to install the following packages.
```bash
pip install -r requirements.txt
```

## Pretrained Models (Segment Anything Model)
You need to download the pretrained models from the following link.
Then, you need to put the model in the following directory.
```bash
mkdir weights
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

## First Frame Mask Generation
First, you need to generate masks at the first frame of the video. You can use the following command to generate which masks you want to use. After generating masks, you can add more masks using the following command.
```bash
sh script/run_add.sh [GPU_ID]
```

## Video Annotation
After selecting masks, you can annotate the video using the following command.
```bash
sh script/run.sh [GPU_ID] [PORT]
```