# Basic
import os
import cv2
import numpy as np
import imageio.v2 as iio
# Torch
import torch
import torch.nn.functional as torch_f
import torchvision.transforms.functional as torchvision_f
import torchvision.models.optical_flow as optical_flow
# Segment Anything
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.amg import calculate_stability_score
# Custom
from app_utils import (
    cycle_consistency_check, backward_warping_mask, compute_ious, 
    random_sampling, segment_frame, compute_iou_numpy, 
    segment_frame_AnnotatedImage, get_pallete,
)

class GlobalData:
    def __init__(
        self,
        video_dir: str=None,
        anno_dir: str=None,
        vis_dir: str=None,
    ):
        self.video_name = None
        # flow
        self.flow = None
        self.flow_inv = None
        self.cycle_error_prev = None
        self.cycle_error_curr = None
        # masks
        self.prev_frame_masks_torch = []
        self.first_frame_masks = []
        self.prev_frame_masks = []
        self.curr_frame_masks = []
        self.selected_points = []
        self.curr_frame_idx = 1
        self.curr_mask_idx = 0
        self.sam_mask_size = 0
        self.iou = 0.0
        self.stability_score = 0.0
        # data paths
        self.frame_paths = []
        # etc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_iter = 5
        self.n_max_masks = 32
        self.make_video = True
        self.fps = 5
        self.selected_mask_idx = -1
        # paths
        self.video_dir = video_dir
        self.anno_dir = anno_dir
        self.vis_dir = vis_dir
        # Segment Anything Model
        self.sam = sam_model_registry['vit_h'](checkpoint="/home/cvlab14/project/seongchan/mug/weights/sam_vit_h_4b8939.pth").to(device=self.device)
        self.sam.eval()
        self.sam_predictor = SamPredictor(self.sam)
        # RAFT Model
        self.raft = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.C_T_SKHT_K_V2).to(device=self.device)
        self.raft_transform = optical_flow.Raft_Large_Weights.C_T_SKHT_V2.transforms()
        # image and mask resize shape
        self.resize_shape = (480, 864)
        # color
        self.pallete, self.hex_pallete = get_pallete()

    def get_hex_pallete(self):
        return self.hex_pallete
    
    def get_random_color_map(self, n_colors=5):
        colors = np.random.choice(list(self.pallete.keys()), n_colors, replace=False)
        color_map = []
        self.color_map = []
        for color in colors:
            self.color_map.append(color)
            color = self.pallete[color]
            color = np.zeros((50, 50, 3), dtype=np.uint8) + color
            color_map.append(color)
        return color_map
    
    def change_color(self, color_idx):
        self.color = self.color_map[color_idx]

    def init(self, video_name, alpha, selected_mask_idx):
        self.video_name = video_name
        # flow
        self.flow = None
        self.flow_inv = None
        self.cycle_error_prev = None
        self.cycle_error_curr = None
        # masks
        self.prev_frame_masks_torch = []
        self.first_frame_masks = []
        self.prev_frame_masks = []
        self.curr_frame_masks = []
        self.selected_points = []
        # etc
        self.curr_frame_idx = 1
        self.curr_mask_idx = 0
        self.sam_mask_size = 0
        self.iou = 0.0
        self.stability_score = 0.0
        self.selected_mask_idx = selected_mask_idx
        # paths
        self.frame_paths = os.listdir(os.path.join(self.video_dir, video_name))
        self.frame_paths.sort()
        # set color
        self.color = np.random.choice(list(self.pallete.keys()))
        # load first frame masks
        mask_dirs = os.listdir(os.path.join(self.anno_dir, video_name))
        mask_dirs.sort()
        if self.selected_mask_idx < 0:
            mask_dirs = mask_dirs
        else:
            mask_dirs = [mask_dirs[self.selected_mask_idx]]
        # load first frame masks and set first frame for Segment Anything
        first_frame = iio.imread(os.path.join(self.video_dir, video_name, self.frame_paths[0]))
        first_frame = cv2.resize(first_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        for mask_idx, mask_dir in enumerate(mask_dirs):
            if self.selected_mask_idx >= 0:
                mask_idx = self.selected_mask_idx
            mask = iio.imread(os.path.join(self.anno_dir, video_name, mask_dir, "000.png"))
            mask = (cv2.resize(mask, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
            self.prev_frame_masks.append(mask.copy())
            self.first_frame_masks.append(mask.copy())
            self.curr_frame_masks.append(np.zeros(self.resize_shape, dtype=np.uint8))
            # save segmented frame
            segmented_frame = segment_frame(first_frame, mask, self.pallete[self.color], alpha)
            os.makedirs(os.path.join(self.vis_dir, video_name, f"track_{mask_idx:03d}"), exist_ok=True)
            iio.imwrite(os.path.join(self.vis_dir, video_name, f"track_{mask_idx:03d}", f"segmented_{self.curr_frame_idx - 1:03d}.png"), segmented_frame)
        # prev_frame_masks to prev_frame_masks_torch
        self.prev_frame_masks_torch = np.stack(self.prev_frame_masks).astype(np.float32)
        self.prev_frame_masks_torch = torch.tensor(self.prev_frame_masks_torch).to(self.device)
        self.n_max_masks = len(mask_dirs)

    @torch.no_grad()
    def get_flow(self):
        prev_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[self.curr_frame_idx - 1]))
        curr_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[self.curr_frame_idx]))
        prev_frame = cv2.resize(prev_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        curr_frame = cv2.resize(curr_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        self.sam_predictor.set_image(curr_frame)
        prev_frame_torch = torchvision_f.to_tensor(prev_frame).unsqueeze(0).to(self.device)
        curr_frame_torch = torchvision_f.to_tensor(curr_frame).unsqueeze(0).to(self.device)
        prev_frame_torch, curr_frame_torch = self.raft_transform(prev_frame_torch, curr_frame_torch)
        # flow estimation
        self.flow = self.raft(prev_frame_torch, curr_frame_torch)[-1].squeeze() # (2, H, W)
        self.flow_inv = self.raft(curr_frame_torch, prev_frame_torch)[-1].squeeze()
        # cycle consistency check
        self.cycle_error_prev = cycle_consistency_check(self.flow, self.flow_inv, device=self.device).reshape(1, *self.resize_shape)
        self.cycle_error_curr = cycle_consistency_check(self.flow_inv, self.flow, device=self.device).reshape(1, *self.resize_shape)

    @torch.no_grad()
    def postprocess_prev_frame_masks_torch(self):
        self.prev_frame_masks_torch = self.prev_frame_masks_torch * self.cycle_error_prev
        self.prev_frame_masks_torch = backward_warping_mask(self.prev_frame_masks_torch, self.flow, self.device)

    @torch.no_grad()
    def automatic_mask_generate(self):
        candidate_masks = []
        for _ in range(self.n_iter):
            point_coords = random_sampling(self.prev_frame_masks_torch[self.curr_mask_idx])
            if point_coords is None:
                break
            point_labels = np.ones(len(point_coords))
            predicted_masks, _, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            stability_scores = calculate_stability_score(
                masks=torch.from_numpy(logits).float(),
                mask_threshold=0.0,
                threshold_offset=1.0,
            )
            if len(predicted_masks[stability_scores > 0.95]) == 0:
                pass
            else:
                candidate_masks.append(predicted_masks[stability_scores > 0.95])
        if len(candidate_masks) == 0:
            self.curr_frame_masks[self.curr_mask_idx] = np.zeros(self.resize_shape, dtype=np.uint8)
            self.iou = 0.0
        else:
            candidate_masks = np.concatenate(candidate_masks)
            candidate_masks_torch = torch.from_numpy(candidate_masks).float().to(self.device)
            candidate_masks_torch = candidate_masks_torch * self.cycle_error_curr
            ious = compute_ious(self.prev_frame_masks_torch[self.curr_mask_idx].unsqueeze(0), candidate_masks_torch)
            max_ious, max_iou_idxs = ious.max(dim=1)
            max_ious, max_iou_idxs = max_ious.cpu().numpy(), max_iou_idxs.cpu().numpy()
            self.curr_frame_masks[self.curr_mask_idx] = candidate_masks[max_iou_idxs[0]].astype(np.uint8)
            self.iou = max_ious[0].item()
    
    def segment_frames(self, alpha):
        # segment curr_frame
        curr_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[self.curr_frame_idx]))
        curr_frame = cv2.resize(curr_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        curr_mask = self.curr_frame_masks[self.curr_mask_idx]
        curr_frame = segment_frame(curr_frame, curr_mask, self.pallete[self.color], alpha)
        # draw points at curr_frame
        point_colors = [(0, 0, 255), (0, 255, 0)]
        for point, label in self.selected_points:
            cv2.circle(curr_frame, point, 5, point_colors[label], -1)
            cv2.circle(curr_frame, point, 5, (0, 0, 0), 2)
        # segment prev_frame
        prev_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[self.curr_frame_idx - 1]))
        prev_frame = cv2.resize(prev_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        prev_mask = self.prev_frame_masks[self.curr_mask_idx]
        prev_frame = segment_frame_AnnotatedImage(prev_frame, prev_mask, self.color)
        # segment first_frame
        first_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[0]))
        first_frame = cv2.resize(first_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        first_mask = self.first_frame_masks[self.curr_mask_idx]
        first_frame = segment_frame_AnnotatedImage(first_frame, first_mask, self.color)
        return curr_frame, prev_frame, first_frame
    
    def is_last_mask(self):
        if self.selected_mask_idx < 0:
            return self.curr_mask_idx >= self.n_max_masks
        else:
            return self.curr_mask_idx >= len(self.first_frame_masks)

    def is_last_frame(self):
        return self.curr_frame_idx >= len(self.frame_paths)
    
    def save_masks(self, alpha):
        curr_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[self.curr_frame_idx]))
        curr_frame = cv2.resize(curr_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
        for mask_idx, mask in enumerate(self.curr_frame_masks):
            if self.selected_mask_idx >= 0:
                mask_idx = self.selected_mask_idx
            iio.imwrite(os.path.join(self.anno_dir, self.video_name, f"track_{mask_idx:03d}", f"{self.curr_frame_idx:03d}.png"), mask * 255)
            segmented_frame = curr_frame.copy()
            segmented_frame = segment_frame(segmented_frame, mask, self.pallete[self.color], alpha)
            iio.imwrite(os.path.join(self.vis_dir, self.video_name, f"track_{mask_idx:03d}", f"segmented_{self.curr_frame_idx:03d}.png"), segmented_frame)
        self.curr_frame_idx += 1
        self.curr_mask_idx = 0
        self.prev_frame_masks = [mask.copy() for mask in self.curr_frame_masks]
        self.curr_frame_masks = [np.zeros(self.resize_shape, dtype=np.uint8) for _ in range(len(self.first_frame_masks))]
        self.selected_points = []
        self.prev_frame_masks_torch = torch.stack([torch.tensor(mask).to(self.device) for mask in self.prev_frame_masks])

    def reject_mask(self):
        self.curr_frame_masks[self.curr_mask_idx] = np.zeros(self.resize_shape, dtype=np.uint8)
        self.selected_points = []
        self.iou = 0.0
    
    @torch.no_grad()
    def previous_mask(self):
        if self.curr_mask_idx > 0:
            self.curr_mask_idx -= 1
            self.selected_points = []
            self.iou = compute_iou_numpy(self.prev_frame_masks[self.curr_mask_idx], self.curr_frame_masks[self.curr_mask_idx])
    
    @torch.no_grad()
    def previous_frame(self):
        if self.curr_frame_idx > 1:
            self.curr_frame_idx -= 1
            self.curr_mask_idx = 0
            self.selected_points = []
            self.iou = 0.0
            self.curr_frame_masks = []
            self.prev_frame_masks = []
            curr_frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[self.curr_frame_idx]))
            curr_frame = cv2.resize(curr_frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
            self.sam_predictor.set_image(curr_frame)
            for mask_idx in range(len(self.first_frame_masks)):
                if self.selected_mask_idx >= 0:
                    mask_idx = self.selected_mask_idx
                mask = (iio.imread(os.path.join(self.anno_dir, self.video_name, f"track_{mask_idx:03d}", f"{self.curr_frame_idx - 1:03d}.png")) > 127).astype(np.uint8)
                self.prev_frame_masks.append(mask)
                mask = (iio.imread(os.path.join(self.anno_dir, self.video_name, f"track_{mask_idx:03d}", f"{self.curr_frame_idx:03d}.png")) > 127).astype(np.uint8)
                self.curr_frame_masks.append(mask)
    
    def get_point(self, point_coords, point_type):
        if point_type == "foreground point":
            self.selected_points.append((point_coords, 1))
        elif point_type == "background point":
            self.selected_points.append((point_coords, 0))
        else:
            self.selected_points.append((point_coords, 1))
    
    def undo_point(self):
        if len(self.selected_points) > 0:
            self.selected_points.pop()

    def change_sam_mask_size(self, sam_mask_size):
        if sam_mask_size == "small":
            self.sam_mask_size = 0
        elif sam_mask_size == "medium":
            self.sam_mask_size = 1
        elif sam_mask_size == "large":
            self.sam_mask_size = 2

    @torch.no_grad()
    def prompt_mask_generate(self):
        if len(self.selected_points) > 0:
            point_coords = np.array([point for point, _ in self.selected_points])
            point_labels = np.array([label for _, label in self.selected_points])
            predicted_masks, _, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            stability_scores = calculate_stability_score(
                masks=torch.from_numpy(logits).float(),
                mask_threshold=0.0,
                threshold_offset=1.0,
            )
            if len(predicted_masks[stability_scores > 0.95]) == 0:
                max_score, max_score_idx = stability_scores.max(dim=0)
                self.curr_frame_masks[self.curr_mask_idx] = predicted_masks[max_score_idx].astype(np.uint8)
                self.stability_score = max_score.item()
            else:
                # self.curr_frame_masks[self.curr_mask_idx] = predicted_masks[stability_scores > 0.95][0].astype(np.uint8)
                if self.sam_mask_size >= len(predicted_masks[stability_scores > 0.95]):
                    sam_mask_size = -1
                else:
                    sam_mask_size = self.sam_mask_size
                self.curr_frame_masks[self.curr_mask_idx] = predicted_masks[stability_scores > 0.95][sam_mask_size].astype(np.uint8)
                self.stability_score = stability_scores[sam_mask_size].item()
            self.iou = compute_iou_numpy(self.prev_frame_masks[self.curr_mask_idx], self.curr_frame_masks[self.curr_mask_idx])
        else:
            self.curr_frame_masks[self.curr_mask_idx] = np.zeros(self.resize_shape, dtype=np.uint8)
            self.iou = 0.0
            self.stability_score = 0.0
    
    def make_segmented_video(self, alpha):
        if not self.make_video:
            return
        os.makedirs(os.path.join(self.vis_dir, self.video_name), exist_ok=True)
        for mask_idx in range(len(self.first_frame_masks)):
            if self.selected_mask_idx >= 0:
                mask_idx = self.selected_mask_idx
            with iio.get_writer(os.path.join(self.vis_dir, self.video_name, f"track_{mask_idx:03d}.mp4"), fps=self.fps) as writer:
                for frame_idx in range(len(self.frame_paths)):
                    frame = iio.imread(os.path.join(self.video_dir, self.video_name, self.frame_paths[frame_idx]))
                    frame = cv2.resize(frame, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
                    mask = iio.imread(os.path.join(self.anno_dir, self.video_name, f"track_{mask_idx:03d}", f"{frame_idx:03d}.png"))
                    mask = cv2.resize(mask, self.resize_shape[::-1], interpolation=cv2.INTER_LINEAR)
                    segmented_frame = segment_frame(frame, mask, self.pallete[self.color], 0.5)
                    writer.append_data(segmented_frame)
    
    def get_info(self):
        if self.video_name is None:
            info_text = [
                ("None", "video"),
                ("000", "frame"),
                ("000", "mask"),
                ("000", "point"),
                ("0.00", "iou"),
                ("0.00", "stability"),
            ]
            info_label = {
                "Frame": 0.0,
                "Mask": 0.0,
            }
        else:
            info_text = [
                (f"{self.video_name}", "video"),
                (f"[{self.curr_frame_idx + 1:03d} / {len(self.frame_paths):03d}]", "frame"),
                (f"[{self.curr_mask_idx + 1:03d} / {len(self.curr_frame_masks):03d}]", "mask"),
                (f"{len(self.selected_points):03d}", "point"),
                (f"{self.iou:.2f}", "iou"),
                (f"{self.stability_score:.2f}", "stability"),
            ]
            info_label = {
                "Frame": (self.curr_frame_idx + 1) / len(self.frame_paths),
                "Mask": (self.curr_mask_idx + 1) / len(self.curr_frame_masks),
            }
        return info_text, info_label