import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import cv2
import time
import imageio.v2 as iio
# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
# PyTorch
import torch
import torch.nn.functional as torch_f
import torchvision.transforms.functional as torchvision_f
import torchvision.models.optical_flow as optical_flow
from torchvision.utils import flow_to_image
# Segment Anything
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import calculate_stability_score, mask_to_rle_pytorch, coco_encode_rle

@torch.no_grad()
def cycle_consistency_check(flow, flow_inv, device='cuda'):
    '''
    flow: (2, H, W), flow[0] is x flow, flow[1] is y flow
    flow_inv: (2, H, W), flow_inv[0] is x flow, flow_inv[1] is y flow
    device: torch.device
    '''
    H, W = flow.shape[1:]
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y, x = y.to(device), x.to(device)
    flow, flow_inv = flow.to(device), flow_inv.to(device)
    # forward flow
    flowed_y = y + flow[1]
    flowed_x = x + flow[0]
    normalized_y = (flowed_y / (H - 1)) * 2 - 1
    normalized_x = (flowed_x / (W - 1)) * 2 - 1
    grid = torch.stack((normalized_x, normalized_y)).permute(1, 2, 0).unsqueeze(0)
    # backward flow
    flow_inv = torch.nn.functional.grid_sample(flow_inv.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    flow_inv = flow_inv.squeeze(0)
    y_cycle = flow[1] + flow_inv[1]
    x_cycle = flow[0] + flow_inv[0]
    # compute cycle consistency error
    cycle_error = torch.sqrt(y_cycle ** 2 + x_cycle ** 2)
    return cycle_error

class EngineV1:
    def __init__(self, config):
        self.gen_anno_dir = config['gen_anno_dir']

        self.fully_gen = config['engine']['fully_gen']
        self.n_iter = config['engine']['n_iter']
        self.n_points_per_mask = config['engine']['n_points_per_mask']
        self.point_per_side = config['engine']['point_per_side']
        self.sampling_method = config['engine']['sampling_method']
        self.device = config['device']
        # Segment Anything Model
        self.sam = sam_model_registry['vit_h'](checkpoint=config['engine']['sam_checkpoint']).to(self.device)
        self.sam.eval()
        # Mask Generator and Predictor (SAM)
        if self.fully_gen:
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=self.point_per_side)
        # SAM Predictor
        self.sam_predictor = SamPredictor(self.sam)
        # RAFT Model
        self.raft = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.C_T_SKHT_V2).to(self.device)
        self.raft.eval()
        self.raft_transforms = optical_flow.Raft_Large_Weights.C_T_SKHT_V2.transforms()

    @torch.no_grad()
    def process_video_clip(self, video_clip):
        # load video clip
        clip_name = video_clip['clip_name']
        frames = video_clip["frames"]
        annos = video_clip.get("annos", None)
        # check dirs
        os.makedirs(os.path.join(self.gen_anno_dir, clip_name, 'warped'), exist_ok=True)
        os.makedirs(os.path.join(self.gen_anno_dir, clip_name, 'origin'), exist_ok=True)
        os.makedirs(os.path.join(self.gen_anno_dir, clip_name, 'abstract'), exist_ok=True)
        # save first frame masks
        assert self.fully_gen, f'Fully generated masks are required for {clip_name}'
        prev_masks = []
        if self.fully_gen:
            # generate masks at first frame with SAM
            annos = self.mask_generator.generate(frames[0])
            for anno in annos:
                prev_masks.append(torch.from_numpy(anno['segmentation']).unsqueeze(0))
        else:
            raise NotImplementedError
        prev_masks = torch.cat(prev_masks, dim=0).to(self.device) # (N, H, W)
        # mask iou non-maximum suppression
        prev_masks = self.mask_iou_nms(prev_masks.float(), iou_threshold=0.7)
        # mask sampling
        n_tracks = 200
        H, W = prev_masks.shape[1:]
        size_threshold = 0.01 * H * W
        if len(prev_masks) > n_tracks:
            print(f"# of objects before sampling: {len(prev_masks)}")
            large_masks, small_masks = [], []
            for mask in prev_masks:
                if mask.sum() > size_threshold:
                    large_masks.append(mask)
                else:
                    small_masks.append(mask)
            n_large_masks = len(large_masks)
            n_small_masks = n_tracks - n_large_masks
            random_indices = np.random.choice(len(small_masks), n_small_masks, replace=False)
            prev_masks = torch.stack(large_masks + [small_masks[i] for i in random_indices], dim=0)
        print(f"# of tracked objects: {len(prev_masks)}")
        # save masks
        self.save_masks(prev_masks, os.path.join(self.gen_anno_dir, clip_name, 'origin', '00000.json'))
        # set point sampling method
        if self.sampling_method == 'random':
            point_sampling = self.random_sampling
        elif self.sampling_method == 'kmeans':
            point_sampling = self.kmeans_sampling
        else:
            raise NotImplementedError
        # iterate over frames
        # print(torch.cuda.memory_allocated() / 1024 ** 3)
        print('start iteration')
        for frame_idx in tqdm(range(1, len(frames))):
            # load frames
            prev_frame = torchvision_f.to_tensor(frames[frame_idx - 1]).to(self.device)
            curr_frame = torchvision_f.to_tensor(frames[frame_idx]).to(self.device)
            prev_frame, curr_frame = self.raft_transforms(prev_frame.unsqueeze(0), curr_frame.unsqueeze(0))
            # flow estimation
            flow = self.raft(prev_frame, curr_frame)[-1].squeeze() # (2, H, W)
            flow_inv = self.raft(curr_frame, prev_frame)[-1].squeeze() # (2, H, W)
            # cycle consistency check
            cycle_error_prev = (cycle_consistency_check(flow, flow_inv, self.device) < 3.0).float().reshape(1, H, W)
            cycle_error_curr = (cycle_consistency_check(flow_inv, flow, self.device) < 3.0).float().reshape(1, H, W)
            # save optical flow visualization and cycle consistency error map
            if frame_idx == 1:
                flow_img = flow_to_image(flow.cpu()).permute(1, 2, 0).numpy()
                iio.imwrite(os.path.join(self.gen_anno_dir, clip_name, 'abstract', 'flow.png'), flow_img)
                flow_inv_img = flow_to_image(flow_inv.cpu()).permute(1, 2, 0).numpy()
                iio.imwrite(os.path.join(self.gen_anno_dir, clip_name, 'abstract', 'flow_inv.png'), flow_inv_img)
                cycle_error_prev_img = Image.fromarray((cycle_error_prev.squeeze().cpu().numpy() * 255).astype(np.uint8))
                cycle_error_prev_img.save(os.path.join(self.gen_anno_dir, clip_name, 'abstract', 'cycle_error.png'))
                cycle_error_curr_img = Image.fromarray((cycle_error_curr.squeeze().cpu().numpy() * 255).astype(np.uint8))
                cycle_error_curr_img.save(os.path.join(self.gen_anno_dir, clip_name, 'abstract', 'cycle_error_inv.png'))
            # update prev_masks - cycle consistency check filtering & backward warping
            prev_masks = prev_masks * cycle_error_prev
            prev_masks = self.backward_warping_mask(prev_masks, flow_inv)
            self.save_masks(prev_masks, os.path.join(self.gen_anno_dir, clip_name, 'warped', f'{frame_idx - 1:05d}_warped.json'))
            # image feature extraction
            self.sam_predictor.set_image(frames[frame_idx])
            # print(torch.cuda.memory_allocated() / 1024 ** 3)
            # mask generation iteration
            curr_masks = []
            curr_points = []
            curr_ious = []
            for mask_idx, prev_mask in enumerate(prev_masks):
                if prev_mask.sum() == 0: # skip empty masks
                    curr_masks.append(torch.zeros(1, H, W).to(self.device))
                    curr_points.append(None)
                    curr_ious.append(np.array([0.0]))
                    continue
                for iou_threshold in [0.9, 0.8, 0.7, 0.6, 0.5]:
                    candidate_masks = []
                    candidate_points = []
                    for _ in range(self.n_iter):
                        # sample points for mask generation (N, 2)
                        sampled_point_coords = point_sampling(prev_mask, n_points=self.n_points_per_mask)
                        sampled_point_labels = np.ones(len(sampled_point_coords))
                        # generate masks with SAM
                        predicted_masks, _, logits = self.sam_predictor.predict(
                            point_coords=sampled_point_coords,
                            point_labels=sampled_point_labels,
                            multimask_output=True,
                        )
                        # mask filtering with stability score
                        stability_scores = calculate_stability_score(
                            masks=torch.from_numpy(logits).float(),
                            mask_threshold=0.0,
                            threshold_offset=1.0,
                        )
                        if len(predicted_masks[stability_scores > 0.95]) == 0: # skip empty masks
                            continue
                        # append masks and points
                        candidate_masks.append(torch.from_numpy(predicted_masks[stability_scores > 0.95]))
                        candidate_points += [sampled_point_coords] * len(predicted_masks[stability_scores > 0.95])
                    if len(candidate_points) == 0:
                        if iou_threshold == 0.5: # skip empty masks
                            curr_masks.append(torch.zeros(1, H, W).to(self.device))
                            curr_points.append(None)
                            curr_ious.append(np.array([0.0]))
                        continue
                    # concatenate masks and points
                    candidate_masks = torch.cat(candidate_masks, dim=0).float().to(self.device)
                    candidate_points = np.stack(candidate_points, axis=0)
                    # compute iou between generated masks and prev frame masks
                    candidate_masks_filtered = candidate_masks * cycle_error_curr
                    ious = self.compute_ious(prev_mask.unsqueeze(0), candidate_masks_filtered)
                    # update prev_masks
                    max_ious, max_ious_indices = ious.max(dim=1)
                    max_ious, max_ious_indices = max_ious.cpu().numpy(), max_ious_indices.cpu().numpy()
                    if max_ious.max() >= iou_threshold or iou_threshold == 0.5:
                        curr_masks.append(candidate_masks[max_ious_indices])
                        curr_points.append(candidate_points[max_ious_indices].squeeze())
                        curr_ious.append(max_ious)
                        break
            # print(torch.cuda.memory_allocated() / 1024 ** 3)
            # save masks
            curr_masks = torch.cat(curr_masks, dim=0)
            curr_ious = np.concatenate(curr_ious, axis=0)
            self.save_masks(curr_masks, os.path.join(self.gen_anno_dir, clip_name, 'origin', f'{frame_idx:05d}.json'), point_coords=curr_points, ious=curr_ious)
            # update prev_frame_masks
            prev_masks = curr_masks.float().to(self.device)
            # clear memory
            del prev_frame, curr_frame, flow, flow_inv, cycle_error_prev, cycle_error_curr, curr_masks, curr_points, curr_ious
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated() / 1024 ** 3)

    def visualization(self, frame, curr_mask, prev_mask, point_coords, flow=None):
        '''
        args:
            frame: np.ndarray, (H, W, 3), frame
            curr_mask: np.ndarray, (H, W), current mask
            prev_mask: torch.Tensor, (H, W), previous mask
            flow: torch.Tensor, (2, H, W), optical flow
            point_coords: np.ndarray, (N, 2), coordinates of points
        '''
        # visualize current mask
        if point_coords is not None:
            curr_y, curr_x = np.where(curr_mask)
            frame[curr_y, curr_x] = frame[curr_y, curr_x] * 0.5 + np.array([255, 0, 0]) * 0.5
            curr_y, curr_x = np.where(cv2.Canny(curr_mask.astype(np.uint8), 0, 1))
            frame[curr_y, curr_x] = np.array([0, 0, 0])
        # visualize previous mask
        prev_y, prev_x = torch.where(prev_mask)
        if flow is not None:
            flowed_y = prev_y + flow[1, prev_y, prev_x]
            flowed_x = prev_x + flow[0, prev_y, prev_x]
            prev_y, prev_x = flowed_y, flowed_x
        prev_y, prev_x = prev_y.cpu().numpy().astype(np.int32), prev_x.cpu().numpy().astype(np.int32)
        frame[prev_y, prev_x] = frame[prev_y, prev_x] * 0.5 + np.array([0, 255, 0]) * 0.5
        # visualize points
        if point_coords is not None:
            for point in point_coords:
                x, y = int(point[0]), int(point[1])
                frame = cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                frame = cv2.circle(frame, (x, y), 5, (0, 0, 0), 2)
        return frame
    
    @torch.no_grad()
    def mask_iou_nms(self, masks, iou_threshold):
        '''
        args:
            masks: torch.Tensor, (N, H, W), binary masks
            iou_threshold: float, iou threshold for non-maximum suppression
        returns:
            selected_masks: torch.Tensor, (M, H, W), binary masks
        '''
        ious = self.compute_ious(masks, masks)
        ious = ious - torch.eye(len(masks)).to(self.device)
        ious = ious > iou_threshold
        pair = torch.where(ious)
        rejected_idx = pair[0][pair[0] > pair[1]].unique().cpu()
        selected_masks = torch.cat([masks[i].unsqueeze(0) for i in range(len(masks)) if i not in rejected_idx], dim=0)
        return selected_masks

    def kmeans_sampling(self, mask, n_points=5):
        '''
        args:
            mask: torch.Tensor, (H, W), binary mask (0 or 1)
            flow: torch.Tensor, (2, H, W), optical flow
            n_points: int, number of points to sample
        returns:
            point_coords: np.ndarray, (n_points, 2), coordinates of sampled points
        '''
        y, x = torch.where(mask)
        points = torch.stack([x, y], dim=1).cpu().numpy()
        if len(points) < n_points:
            return points
        # sample points
        kmeans = KMeans(n_clusters=n_points, random_state=0, n_init='auto').fit(points)
        return kmeans.cluster_centers_
    
    def random_sampling(self, mask, n_points=5):
        '''
        args:
            mask: torch.Tensor, (H, W), binary mask (0 or 1)
            flow: torch.Tensor, (2, H, W), optical flow
            n_points: int, number of points to sample
        returns:
            point_coords: np.ndarray, (n_points, 2), coordinates of sampled points
        '''
        y, x = torch.where(mask)
        points = torch.stack([x, y], dim=1).cpu().numpy()
        if len(points) < n_points:
            return points
        # sample points
        indices = np.random.choice(len(points), n_points, replace=False)
        return points[indices]

    @torch.no_grad()
    def compute_ious(self, masks1, masks2):
        '''
        args:
            masks1: torch.Tensor, (N1, H, W), binary masks
            masks2: torch.Tensor, (N2, H, W), binary masks
        returns:
            ious: torch.Tensor, (N1, N2), intersection over union
        '''
        N1, N2 = masks1.shape[0], masks2.shape[0]
        masks1 = masks1.view(N1, -1)
        masks2 = masks2.view(N2, -1)
        intersection = masks1 @ masks2.T
        union = masks1.sum(dim=1)[:, None] + masks2.sum(dim=1)[None, :] - intersection
        ious = intersection / union
        return ious
    
    @torch.no_grad()
    def backward_warping_mask(self, masks, flow):
        '''
        args:
            masks: torch.Tensor, (N, H, W), binary masks
            flow: torch.Tensor, (H, W, 2), optical flow
        returns:
            warped_masks: torch.Tensor, (N, H, W), binary masks
        '''
        _, H, W = masks.shape
        masks, flow = masks.to(self.device), flow.to(self.device)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.to(self.device), x.to(self.device)
        y = y + flow[1]
        x = x + flow[0]
        y = (y / (H - 1)) * 2 - 1
        x = (x / (W - 1)) * 2 - 1
        grid = torch.stack((x, y)).permute(1, 2, 0).unsqueeze(0)
        masks = masks.unsqueeze(0)
        warped_masks = torch_f.grid_sample(masks, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
        return (warped_masks > 0.5) * 1.0
    
    def save_masks(self, masks, save_path, point_coords=None, ious=None):
        '''
        args:
            annos: list of dict, list of annotations
            save_path: str, path to save the annotations
            frame_idx: int, frame index
        '''
        masks = mask_to_rle_pytorch(masks.cpu().bool())
        ious = ious if ious is not None else np.ones(len(masks))
        if point_coords is None:
            point_coords = [None] * len(masks)
        else:
            for i in range(len(point_coords)):
                if point_coords[i] is not None:
                    point_coords[i] = point_coords[i].tolist()
        annos = []
        ious = ious.astype(np.float64).tolist()
        for mask_id, mask in enumerate(masks):
            anno = coco_encode_rle(mask)
            anno['id'] = mask_id
            anno['point_coords'] = point_coords[mask_id]
            anno['iou'] = ious[mask_id]
            annos.append(anno)
        with open(save_path, 'w') as f:
            json.dump(annos, f)