import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import cv2
import time
import imageio.v2 as iio
from sklearn_extra.cluster import KMedoids
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
    # print(f'grid shape: {grid.shape}') 
    # backward flow
    flow_inv = torch.nn.functional.grid_sample(flow_inv.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    flow_inv = flow_inv.squeeze(0)
    y_cycle = flow[1] + flow_inv[1]
    x_cycle = flow[0] + flow_inv[0]
    # compute cycle consistency error
    cycle_error = torch.sqrt(y_cycle ** 2 + x_cycle ** 2)
    return cycle_error

class EngineV1:
    def __init__(self, config, device):
        self.fully_gen = config['engine']['fully_gen']
        self.gen_anno_dir = config['gen_anno_dir']
        self.device = device
        self.is_vis = True
        # Segment Anything Model
        self.sam = sam_model_registry['vit_h'](checkpoint='/home/cvlab11/project/kschan/models/sam/sam_vit_h_4b8939.pth').to(device)
        self.sam.eval()
        # Mask Generator and Predictor (SAM)
        if self.fully_gen:
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=32)
        # SAM Predictor
        self.sam_predictor = SamPredictor(self.sam)
        # RAFT Model
        self.raft = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.C_T_SKHT_V2).to(device)
        self.raft.eval()
        self.raft_transforms = optical_flow.Raft_Large_Weights.C_T_SKHT_V2.transforms()
        self.n_iter = config['engine']['n_iter']
        self.n_points_per_mask = config['engine']['n_points_per_mask']
        self.vis_dir = config['vis_dir']

    @torch.no_grad()
    def process_video_clip(self, video_clip):
        clip_name = video_clip['clip_name']
        frames = video_clip["frames"]
        annos = video_clip.get("annos", None)
        # save first frame masks
        prev_frame_masks = []
        gen_annos = []
        assert self.fully_gen, f'Fully generated masks are required for {clip_name}'
        H, W = frames[0].shape[0], frames[0].shape[1]
        mask_size_threshold = H * W * 0.02
        if self.fully_gen:
            # generate masks at first frame with SAM
            annos = self.mask_generator.generate(frames[0])
            # if len(annos) > 40:
            #     l_annos = [anno for anno in annos if anno['area'] >= mask_size_threshold]
            #     s_annos = [anno for anno in annos if anno['area'] < mask_size_threshold]
            #     print(f'# of large masks: {len(l_annos)}')
            #     print(f'# of small masks: {len(s_annos)}')
            #     # print(f'Number of masks generated: {len(annos)}')
            #     n_s_annos = max(1, 40 - len(l_annos))
            #     s_annos = np.random.choice(s_annos, n_s_annos, replace=False)
            #     annos = l_annos + list(s_annos)
            print(f'# of selected masks: {len(annos)}')
            # save annos
            for mask_id, anno in enumerate(annos):
                prev_frame_mask = torch.from_numpy(anno['segmentation']).unsqueeze(0)
                prev_frame_masks.append(prev_frame_mask)
                gen_anno = coco_encode_rle(mask_to_rle_pytorch(prev_frame_mask)[0])
                gen_anno['point_coords'] = anno['point_coords']
                gen_anno['id'] = mask_id
                gen_anno['iou'] = 1.0
                gen_annos.append(gen_anno)
        else:
            # save annos
            for mask_id, anno in enumerate(annos):
                prev_frame_mask = torch.from_numpy(anno['counts']).unsqueeze(0)
                prev_frame_masks.append(prev_frame_mask)
                gen_anno = coco_encode_rle(mask_to_rle_pytorch(prev_frame_mask)[0])
                gen_anno['id'] = mask_id
                gen_annos.append(gen_anno)
        os.makedirs(os.path.join(self.gen_anno_dir, clip_name), exist_ok=True)
        with open(os.path.join(self.gen_anno_dir, clip_name, '00000.json'), 'w') as f:
            json.dump(gen_annos, f)
        prev_frame_masks = torch.cat(prev_frame_masks, dim=0) # (N, H, W)
        H, W = frames.shape[1], frames.shape[2]
        # generate masks after first frame
        segmented_frames = {}
        for mask_idx, mask in enumerate(prev_frame_masks):
            y, x = np.where(mask.numpy())
            segmented_frame = frames[0].copy()
            segmented_frame[y, x] = segmented_frame[y, x] * 0.5 + np.array([255, 0, 0]) * 0.5
            # segmented_frame = np.concatenate([segmented_frame, segmented_frame, segmented_frame], axis=1)
            segmented_frames[mask_idx] = [segmented_frame]
        prev_frame_masks = prev_frame_masks.to(self.device)
        # iterate over frames
        print('start iteration')
        if self.is_vis:
            os.makedirs(f'{self.vis_dir}/{clip_name}', exist_ok=True)
        # os.makedirs(f'/home/cvlab11/project/kschan/a_code/videosam-data_engine/results/{clip_name}', exist_ok=True)
        flow2img = True
        for frame_idx in tqdm(range(1, len(frames))):
            gen_annos = []
            # load frames
            prev_frame = torchvision_f.to_tensor(frames[frame_idx - 1]).to(self.device)
            curr_frame = torchvision_f.to_tensor(frames[frame_idx]).to(self.device)
            prev_frame, curr_frame = self.raft_transforms(prev_frame.unsqueeze(0), curr_frame.unsqueeze(0))
            # flow estimation
            flow = self.raft(prev_frame, curr_frame)[-1].squeeze() # (2, H, W)
            flow_inv = self.raft(curr_frame, prev_frame)[-1].squeeze() # (2, H, W)
            if flow2img and self.is_vis:
                flow_img = flow_to_image(flow.unsqueeze(0))
                flow_img = flow_img.squeeze().permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(f'{self.vis_dir}/{clip_name}/flow-{clip_name}-{frame_idx:05d}.png', flow_img)
                # cv2.imwrite(f'/home/cvlab11/project/kschan/a_code/videosam-data_engine/results/{clip_name}/flow-{clip_name}-{frame_idx:05d}.png', flow_img)
                flow_inv_img = flow_to_image(flow_inv.unsqueeze(0))
                flow_inv_img = flow_inv_img.squeeze().permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(f'{self.vis_dir}/{clip_name}/flow_inv-{clip_name}-{frame_idx:05d}.png', flow_inv_img)
                # cv2.imwrite(f'/home/cvlab11/project/kschan/a_code/videosam-data_engine/results/{clip_name}/flow_inv-{clip_name}-{frame_idx:05d}.png', flow_inv_img)
                flow2img = False
            # cycle consistency check
            cycle_error_prev = (cycle_consistency_check(flow, flow_inv, self.device) < 3.0).float().reshape(1, H, W)
            cycle_error_curr = (cycle_consistency_check(flow_inv, flow, self.device) < 3.0).float().reshape(1, H, W)
            prev_frame_masks = prev_frame_masks * cycle_error_prev
            # mask generation iteration
            masks = []
            points = []
            ious = []
            self.sam_predictor.set_image(frames[frame_idx])
            sampling_method = 'random' # 'random' or 'kmeans'
            if sampling_method == 'random':
                point_sampling = self.random_sampling
            elif sampling_method == 'kmeans':
                point_sampling = self.kmeans_sampling
            else:
                raise NotImplementedError
            for mask_idx, prev_frame_mask in enumerate(prev_frame_masks):
                # t = time.time()
                masks_per_mask = []
                points_per_mask = []
                for _ in range(self.n_iter):
                    if len(points_per_mask) > 3:
                        break
                    # sample points for mask generation (N, 2)
                    sampled_point_coords = point_sampling(prev_frame_mask, flow, n_points=self.n_points_per_mask)
                    if sampled_point_coords is None:
                        break
                    sampled_point_labels = np.ones(len(sampled_point_coords)) # (N,)
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
                    if len(predicted_masks[stability_scores > 0.95]) == 0:
                        continue
                    masks_per_mask.append(torch.from_numpy(predicted_masks[stability_scores > 0.95]))
                    points_per_mask += [sampled_point_coords] * len(predicted_masks[stability_scores > 0.95])
                if len(points_per_mask) == 0:
                    masks.append(torch.zeros(1, H, W))
                    points.append(None)
                    ious.append(np.array([0.0]))
                else:
                    masks_per_mask = torch.concat(masks_per_mask, axis=0)
                    points_per_mask = np.stack(points_per_mask, axis=0)
                    # compute iou between generated masks and prev frame masks
                    curr_frame_masks = masks_per_mask.float().to(self.device) * cycle_error_curr
                    prev_frame_mask = self.backward_warping_mask(prev_frame_mask.unsqueeze(0), flow_inv)
                    ious_ = self.compute_ious(prev_frame_mask, curr_frame_masks)
                    # update prev_frame_masks
                    max_ious, max_ious_indices = ious_.max(dim=1)
                    max_ious, max_ious_indices = max_ious.cpu().numpy(), max_ious_indices.cpu().numpy()
                    masks.append(masks_per_mask[max_ious_indices]) # fix this
                    points.append(points_per_mask[max_ious_indices].squeeze())
                    ious.append(max_ious)
                # visualization
                if self.is_vis:
                    segmented_frame = self.visualization(
                        frames[frame_idx].copy(),
                        masks[-1].squeeze().numpy(), 
                        prev_frame_mask.squeeze().clone().detach(), 
                        points[-1], 
                    )
                    segmented_frames[mask_idx].append(segmented_frame)
                # print(f'mask generation time: {time.time() - t}')
            # masks = np.concatenate(masks, axis=0)
            masks = torch.cat(masks, dim=0).cpu().numpy()
            ious = np.concatenate(ious, axis=0)
            # update prev_frame_masks
            # prev_frame_masks = torch.from_numpy(masks).float().to(self.device)
            prev_frame_masks = []
            # save annos
            for mask_idx, (mask, point_coord) in enumerate(zip(masks, points)):
                mask = torch.from_numpy(mask).unsqueeze(0).bool()
                prev_frame_masks.append(mask)
                gen_anno = coco_encode_rle(mask_to_rle_pytorch(mask)[0])
                gen_anno['point_coords'] = [] if point_coord is None else point_coord.tolist()
                gen_anno['id'] = mask_idx
                gen_anno['iou'] = 0.0 if point_coord is None else ious[mask_idx].item()
                gen_annos.append(gen_anno)
                
            prev_frame_masks = torch.cat(prev_frame_masks, dim=0).float().to(self.device)
            os.makedirs(os.path.join(self.gen_anno_dir, clip_name), exist_ok=True)
            with open(os.path.join(self.gen_anno_dir, clip_name, f'{frame_idx:05d}.json'), 'w') as f:
                json.dump(gen_annos, f)
        if self.is_vis:
            # save segmented frames
            for mask_idx, segmented_frames_ in segmented_frames.items():
                writer = iio.get_writer(f'{self.vis_dir}/{clip_name}/segmented-{clip_name}-m{mask_idx}.mp4', fps=5)
                # writer = iio.get_writer(f'/home/cvlab11/project/kschan/a_code/videosam-data_engine/results/{clip_name}/flow-{clip_name}-multi-m{mask_idx}.mp4', fps=5)
                for segmented_frame in segmented_frames_:
                    segmented_frame = cv2.resize(segmented_frame, (1088, 688))
                    writer.append_data(segmented_frame)
                writer.close()
    
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

    def kmeans_sampling(self, mask, flow, n_points=5):
        '''
        args:
            mask: torch.Tensor, (H, W), binary mask (0 or 1)
            flow: torch.Tensor, (2, H, W), optical flow
            n_points: int, number of points to sample
        returns:
            point_coords: np.ndarray, (n_points, 2), coordinates of sampled points
        '''
        y, x = torch.where(mask)
        # flowed points
        flowed_y = y + flow[1, y, x]
        flowed_x = x + flow[0, y, x]
        points = torch.stack([flowed_x, flowed_y], dim=1).cpu().numpy()
        if len(points) < n_points:
            return None
        # sample points
        kmeans = KMeans(n_clusters=n_points, random_state=0, n_init='auto').fit(points)
        return kmeans.cluster_centers_
    
    def random_sampling(self, mask, flow, n_points=5):
        '''
        args:
            mask: torch.Tensor, (H, W), binary mask (0 or 1)
            flow: torch.Tensor, (2, H, W), optical flow
            n_points: int, number of points to sample
        returns:
            point_coords: np.ndarray, (n_points, 2), coordinates of sampled points
        '''
        y, x = torch.where(mask)
        # flowed points
        flowed_y = y + flow[1, y, x]
        flowed_x = x + flow[0, y, x]
        points = torch.stack([flowed_x, flowed_y], dim=1).cpu().numpy()
        if len(points) < n_points:
            return None
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
    def backward_warping_mask(self, masks, flow): # TODO: implement this
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