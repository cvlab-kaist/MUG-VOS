# Multi-Granularity Video Object Segmentation (MUG-VOS)

## Installation
First, you need to install the following packages.
```bash
pip install -r requirements.txt
```

## Pretrained Model Weights (Segment Anything Model)
You need to download the pretrained models from the following link.
Then, you need to put the model in the following directory.
```bash
mkdir weights
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

## MUG-VOS Data Annotation Tool
This project is to utilize Segment Anything Model (SAM) to annotate video data.
As well as Segment Anything Model (SAM), we also use RAFT, optical flow model, to track objects in the video.
```bash
cd mug_vos_anno_tool
```

### First Frame Mask Generation
First, you need to generate masks at the first frame of the video. You can use the following command to generate which masks you want to use.
```bash
sh script/run_add.sh [GPU_ID]
```

### Video Annotation
After generating masks, you can annotate the video using the following command.
```bash
sh script/run.sh [GPU_ID] [PORT]
```

## MUG-VOS Data Collection Pipeline
You can also use the following command to generate masks automatically. This data collection pipeline use SAM and RAFT to generate masks and track objects in the video.
```bash
cd mug_vos_dataengine
sh run.sh [GPU_ID]
```