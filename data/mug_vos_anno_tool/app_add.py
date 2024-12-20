import os
import cv2
import argparse
import numpy as np
import gradio as gr
import imageio.v2 as iio
# Pytorch
import torch
# Segment Anything
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.amg import calculate_stability_score
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--video_dir", type=str, default="test")
parser.add_argument("--anno_dir", type=str, default="test")
parser.add_argument("--vis_dir", type=str, default="test")
parser.add_argument("--max_masks", type=int, default=2)
args = parser.parse_args()
# check arguments
assert args.port != 7860, "Please specify the port number"
assert args.video_dir != "test", "Please specify the video directory"
assert args.anno_dir != "test", "Please specify the annotation directory"
assert args.vis_dir != "test", "Please specify the visualization directory"
# global variables
dir_dict = {
    "video_dir": args.video_dir,
    "anno_dir": args.anno_dir,
    "vis_dir": args.vis_dir,
}
max_masks = args.max_masks
colors = np.random.randint(0, 255, (max_masks, 3))
# Segment Anything Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_h"](checkpoint="../weights/sam_vit_h_4b8939.pth").to(device=device)
sam.eval()
sam_predictor = SamPredictor(sam)
# main function
def main(args):
    video_names = os.listdir(dir_dict["video_dir"])
    video_names.sort()
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                Info = gr.Textbox("Video: None | Frame: 000 | Mask: 000 | Point: 000")
                selected_points = gr.State(value=[])
                segmented_frame = gr.Image(type="numpy", show_label=False, show_download_button=False)
                with gr.Column():
                    with gr.Row():
                        undo_point_button = gr.Button("Undo point")
                        accept_mask_button = gr.Button("Accept mask")
                    point_type = gr.Radio(["Foreground", "Background"], label="Point type")
                video_name = gr.Dropdown(
                    choices=video_names,
                    label="Video name",
                    info="Select a video to annotate"
                )
        # 1. Select video to annotate
        video_name.change(
            fn=select_video,
            inputs=[video_name],
            outputs=[segmented_frame, selected_points, Info],
        )
        # 2. Select point
        segmented_frame.select(
            fn=select_point,
            inputs=[video_name, selected_points, point_type],
            outputs=[segmented_frame, selected_points, Info],
        )
        # 3. Undo point
        undo_point_button.click(
            fn=undo_point,
            inputs=[video_name, selected_points],
            outputs=[segmented_frame, selected_points, Info],
        )
        # 4. Accept mask
        accept_mask_button.click(
            fn=accept_mask,
            inputs=[video_name, selected_points],
            outputs=[segmented_frame, selected_points, Info],
        )
    # Launch the interface
    demo.queue().launch(debug=True, server_port=args.port)

def select_video(video_name):
    global curr_masks
    # get first frame and set image
    frame_paths = os.listdir(os.path.join(dir_dict["video_dir"], video_name))
    frame_paths.sort()
    frame_path = os.path.join(dir_dict["video_dir"], video_name, frame_paths[0])
    frame = iio.imread(frame_path)
    H, W = frame.shape[:2]
    sam_predictor.set_image(frame)
    # get masks
    curr_masks = []
    os.makedirs(os.path.join(dir_dict["anno_dir"], video_name), exist_ok=True)
    curr_mask_dirs = os.listdir(os.path.join(dir_dict["anno_dir"], video_name))
    for mask_dir in curr_mask_dirs:
        mask = (iio.imread(os.path.join(dir_dict["anno_dir"], video_name, mask_dir, "000.png")) > 0).astype(np.uint8)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        curr_masks.append(mask)
    curr_masks = sorted(curr_masks, key=lambda x: -np.sum(x))
    # get segmented frame
    if len(curr_masks) >= max_masks:
        segmented_frame = frame.copy()
        for mask_idx, mask in enumerate(curr_masks):
            coords = np.where(mask)
            segmented_frame[coords] = colors[mask_idx] * 0.5 + segmented_frame[coords] * 0.5
            edge = cv2.Canny(mask, 0, 1)
            coords = np.where(edge)
            segmented_frame[coords] = np.array([0, 0, 0])
        iio.imwrite(os.path.join(dir_dict["vis_dir"], f"{video_name}.png"), segmented_frame)
        print(f"Annotation Done for Video: {video_name}")
        return (None, [], f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: 000")
    curr_masks.append(np.zeros((H, W), dtype=np.uint8))
    # get segmented frame
    segmented_frame = frame.copy()
    for mask_idx, mask in enumerate(curr_masks):
        mask = mask.astype(np.uint8)
        coords = np.where(mask)
        segmented_frame[coords] = colors[mask_idx] * 0.7 + segmented_frame[coords] * 0.3
        edge = cv2.Canny(mask, 0, 1)
        coords = np.where(edge)
        segmented_frame[coords] = np.array([0, 0, 0])
    segmented_frame = np.concatenate([frame, segmented_frame], axis=1)
    print(f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: 000")
    return segmented_frame, [], f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: 000"

def select_point(video_name, selected_points, point_type, evt: gr.SelectData):
    global curr_masks
    curr_masks = curr_masks[:-1]
    # get point
    if point_type == "Foreground":
        selected_points.append((evt.index, 1))
    elif point_type == "Background":
        selected_points.append((evt.index, 0))
    else:
        selected_points.append((evt.index, 1))
    # segment anything with points
    point_coords = np.array([point for point, _ in selected_points])
    point_labels = np.array([label for _, label in selected_points])
    # get frame
    frame_paths = os.listdir(os.path.join(dir_dict["video_dir"], video_name))
    frame_paths.sort()
    frame_path = os.path.join(dir_dict["video_dir"], video_name, frame_paths[0])
    frame = iio.imread(frame_path)
    # get mask
    predicted_masks, _, logits = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    # get stability score
    stability_score = calculate_stability_score(
        masks=torch.from_numpy(logits).float(),
        mask_threshold=0.0,
        threshold_offset=1.0,
    )
    if len(predicted_masks[stability_score > 0.95]) == 0:
        max_score, max_score_idx = stability_score.max(dim=0)
        curr_mask = predicted_masks[max_score_idx].astype(np.uint8)
    else:
        curr_mask = predicted_masks[stability_score > 0.95][0].astype(np.uint8)
    # compute ious with previous masks
    is_overlap = False
    for mask in curr_masks:
        iou = np.sum(mask & curr_mask) / np.sum(mask | curr_mask)
        if iou > 0.7:
            is_overlap = True
            break
    curr_masks.append(curr_mask)
    # get segmented frame
    segmented_frame = frame.copy()
    for mask_idx, mask in enumerate(curr_masks):
        mask = mask.astype(np.uint8)
        coords = np.where(mask)
        segmented_frame[coords] = colors[mask_idx] * 0.7 + segmented_frame[coords] * 0.3
        edge = cv2.Canny(mask, 0, 1)
        coords = np.where(edge)
        segmented_frame[coords] = np.array([0, 0, 0])
    segmented_frame = np.concatenate([frame, segmented_frame], axis=1)
    coords = np.where(curr_mask)
    if is_overlap:
        segmented_frame[coords] = np.array([255, 0, 0])
    else:
        segmented_frame[coords] = colors[len(curr_masks) - 1] * 0.7 + segmented_frame[coords] * 0.3
    edge = cv2.Canny(curr_mask, 0, 1)
    coords = np.where(edge)
    segmented_frame[coords] = np.array([0, 0, 0])
    point_colors = [(0, 0, 255), (0, 255, 0)]
    for point, label in selected_points:
        cv2.circle(segmented_frame, point, 5, point_colors[label], -1)
        cv2.circle(segmented_frame, point, 5, (0, 0, 0), 2)
    print(f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: {len(selected_points):03d}")
    return segmented_frame, selected_points, f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: {len(selected_points):03d}"

def undo_point(video_name, selected_points):
    global curr_masks
    curr_masks = curr_masks[:-1]
    # get frame
    frame_paths = os.listdir(os.path.join(dir_dict["video_dir"], video_name))
    frame_paths.sort()
    frame_path = os.path.join(dir_dict["video_dir"], video_name, frame_paths[0])
    frame = iio.imread(frame_path)
    H, W = frame.shape[:2]
    if len(selected_points) > 0:
        selected_points.pop()
    if len(selected_points) > 0:
        # get point
        point_coords = np.array([point for point, _ in selected_points])
        point_labels = np.array([label for _, label in selected_points])
        # get mask
        predicted_masks, _, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # get stability score
        stability_score = calculate_stability_score(
            masks=torch.from_numpy(logits).float(),
            mask_threshold=0.0,
            threshold_offset=1.0,
        )
        if len(predicted_masks[stability_score > 0.95]) == 0:
            max_score, max_score_idx = stability_score.max(dim=0)
            curr_mask = predicted_masks[max_score_idx].astype(np.uint8)
        else:
            curr_mask = predicted_masks[stability_score > 0.95][0].astype(np.uint8)
    else:
        curr_mask = np.zeros((H, W), dtype=np.uint8)
    # compute ious with previous masks
    is_overlap = False
    for mask in curr_masks:
        iou = np.sum(mask & curr_mask) / np.sum(mask | curr_mask)
        if iou > 0.7:
            is_overlap = True
            break
    curr_masks.append(curr_mask)
    # get segmented frame
    segmented_frame = frame.copy()
    for mask_idx, mask in enumerate(curr_masks):
        mask = mask.astype(np.uint8)
        coords = np.where(mask)
        segmented_frame[coords] = colors[mask_idx] * 0.7 + segmented_frame[coords] * 0.3
        edge = cv2.Canny(mask, 0, 1)
        coords = np.where(edge)
        segmented_frame[coords] = np.array([0, 0, 0])
    segmented_frame = np.concatenate([frame, segmented_frame], axis=1)
    coords = np.where(curr_mask)
    if is_overlap:
        segmented_frame[coords] = np.array([255, 0, 0])
    else:
        segmented_frame[coords] = colors[len(curr_masks) - 1] * 0.7 + segmented_frame[coords] * 0.3
    edge = cv2.Canny(curr_mask, 0, 1)
    coords = np.where(edge)
    segmented_frame[coords] = np.array([0, 0, 0])
    point_colors = [(0, 0, 255), (0, 255, 0)]
    for point, label in selected_points:
        cv2.circle(segmented_frame, point, 5, point_colors[label], -1)
        cv2.circle(segmented_frame, point, 5, (0, 0, 0), 2)
    print(f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: {len(selected_points):03d}")
    return segmented_frame, selected_points, f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: {len(selected_points):03d}"

def accept_mask(video_name, selected_points):
    global curr_masks
    # save masks
    frame_paths = os.listdir(os.path.join(dir_dict["video_dir"], video_name))
    frame_paths.sort()
    segmented_frame = iio.imread(os.path.join(dir_dict["video_dir"], video_name, frame_paths[0]))
    if len(curr_masks) >= max_masks:
        for mask_idx, mask in enumerate(curr_masks):
            mask_dir = os.path.join(dir_dict["anno_dir"], video_name, f"track_{mask_idx:03d}")
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, f"000.png")
            iio.imwrite(mask_path, mask * 255)
            coords = np.where(mask)
            segmented_frame[coords] = colors[mask_idx] * 0.5 + segmented_frame[coords] * 0.5
            edge = cv2.Canny(mask, 0, 1)
            coords = np.where(edge)
            segmented_frame[coords] = np.array([0, 0, 0])
        iio.imwrite(os.path.join(dir_dict["vis_dir"], f"{video_name}.png"), segmented_frame)
        print(f"Annotation Done for Video: {video_name}")
        return (None, None, None)
    else:
        H, W = curr_masks[0].shape
        # curr_masks = sorted(curr_masks, key=lambda x: -np.sum(x))
        curr_masks.append(np.zeros((H, W), dtype=np.uint8))
        frame = segmented_frame.copy()
        for mask_idx, mask in enumerate(curr_masks):
            mask = mask.astype(np.uint8)
            coords = np.where(mask)
            segmented_frame[coords] = colors[mask_idx] * 0.7 + segmented_frame[coords] * 0.3
            edge = cv2.Canny(mask, 0, 1)
            coords = np.where(edge)
            segmented_frame[coords] = np.array([0, 0, 0])
        segmented_frame = np.concatenate([frame, segmented_frame], axis=1)
        print(f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: {len(selected_points)}")
        return segmented_frame, [], f"Video: {video_name} | Mask: {len(curr_masks) - 1:03d} | Point: {len(selected_points)}"

if __name__ == "__main__":
    main(args)