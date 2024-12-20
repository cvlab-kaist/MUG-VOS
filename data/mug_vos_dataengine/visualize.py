
import os
import cv2
import json
import numpy as np
import imageio.v2 as iio
import pycocotools.mask as mask_utils
from tqdm.auto import tqdm

anno_root = '/home/cvlab14/project/seongchan/videosam-data_engine_output/DAVIS_val'
video_root = '/home/cvlab14/project/seongchan/DAVIS/JPEGImages/Full-Resolution'
vis_root = '/home/cvlab14/project/seongchan/videosam-data_engine/vis'
video_list = os.listdir(anno_root)
video_list.sort()

def decode_rle(encoded_rle):
    encoded_rle["counts"] = encoded_rle["counts"].encode("utf-8")
    return {
        "id": encoded_rle["id"],
        "mask": mask_utils.decode(encoded_rle),
        "points": encoded_rle["point_coords"],
        "iou": encoded_rle["iou"],
    }

def visualize(frame, mask, prev_mask, point_coords, prev_point_coords, iou, alpha=0.7):
    visualized_frame = frame.copy()
    # visualize current mask
    y, x = np.where(mask)
    visualized_frame[y, x] = np.array([255, 0, 0]) * alpha + visualized_frame[y, x] * (1 - alpha)
    edge = cv2.Canny(mask.astype(np.uint8), 0, 1)
    y, x = np.where(edge)
    visualized_frame[y, x] = np.array([0, 0, 0])
    # visualize previous mask
    if prev_mask is not None:
        y, x = np.where(prev_mask)
        visualized_frame[y, x] = np.array([0, 0, 255]) * alpha + visualized_frame[y, x] * (1 - alpha)
        edge = cv2.Canny(prev_mask.astype(np.uint8), 0, 1)
        y, x = np.where(edge)
        visualized_frame[y, x] = np.array([0, 0, 0])
    # visualize current point
    if point_coords is not None:
        for point in point_coords:
            if point is None:
                continue
            if not isinstance(point, list):
                continue
            cv2.circle(visualized_frame, (point[0], point[1]), 4, (0, 255, 0), -1)
            cv2.circle(visualized_frame, (point[0], point[1]), 4, (0, 0, 0), 1)
    # visualize previous point
    if prev_point_coords is not None:
        for point in prev_point_coords:
            if point is None:
                continue
            cv2.circle(visualized_frame, (point[0], point[1]), 4, (255, 255, 0), -1)
            cv2.circle(visualized_frame, (point[0], point[1]), 4, (0, 0, 0), 1)
    # visualize iou
    cv2.putText(visualized_frame, f'IoU: {iou:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return visualized_frame

for video in video_list:
    print(f"Processing video {video} ...")
    # load frames and annotations
    frame_list = os.listdir(os.path.join(video_root, video))
    frame_list.sort()
    annos_list = os.listdir(os.path.join(anno_root, video, 'origin'))
    annos_list.sort()
    warped_annos_list = os.listdir(os.path.join(anno_root, video, 'warped'))
    warped_annos_list.sort()
    n_frames = len(frame_list)
    for frame_idx in tqdm(range(n_frames)):
        # load frame
        frame = iio.imread(os.path.join(video_root, video, frame_list[frame_idx]))
        frame = cv2.resize(frame, (856, 480), interpolation=cv2.INTER_LINEAR)
        # load annotations
        with open(os.path.join(anno_root, video, 'origin', annos_list[frame_idx]), 'r') as f:
            annos = json.load(f)
        annos = [decode_rle(anno) for anno in annos]
        masks = [anno['mask'] for anno in annos]
        points = [anno['points'] for anno in annos]
        ious = [anno['iou'] for anno in annos]
        # load warped annotations
        if frame_idx > 0:
            with open(os.path.join(anno_root, video, 'warped', warped_annos_list[frame_idx - 1]), 'r') as f:
                warped_annos = json.load(f)
            warped_annos = [decode_rle(anno) for anno in warped_annos]
            prev_masks = [anno['mask'] for anno in warped_annos]
            prev_points = [anno['points'] for anno in warped_annos]
        else:
            prev_masks = None
            prev_points = None
            visualized_frames = {mask_idx: [] for mask_idx in range(len(masks))}
        # visualize
        for mask_idx in range(len(masks)):
            mask = masks[mask_idx]
            point = points[mask_idx]
            iou = ious[mask_idx]
            if frame_idx > 0:
                prev_mask = prev_masks[mask_idx]
                prev_point = prev_points[mask_idx]
                visualized_frames[mask_idx].append(visualize(frame, mask, prev_mask, point, prev_point, iou))
            else:
                visualized_frames[mask_idx].append(visualize(frame, mask, None, point, None, iou))
    # save visualized frames as video
    os.makedirs(os.path.join(vis_root, video), exist_ok=True)
    for mask_idx, visualized_frame_list in tqdm(visualized_frames.items()):
        visualized_frame_path = os.path.join(vis_root, video, f'visualized_mask_{mask_idx}.mp4')
        visualized_frame_list = [cv2.resize(visualized_frame, (864, 480), interpolation=cv2.INTER_LINEAR) for visualized_frame in visualized_frame_list]
        iio.mimwrite(visualized_frame_path, visualized_frame_list, fps=5)