# Basic
import cv2
import numpy as np
# Torch
import torch
import torch.nn.functional as torch_f

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
    return (cycle_error < 3.0).float()

@torch.no_grad()
def backward_warping_mask(masks, flow, device):
    '''
    args:
        masks: torch.Tensor, (N, H, W), binary masks
        flow: torch.Tensor, (H, W, 2), optical flow
    returns:
        warped_masks: torch.Tensor, (N, H, W), binary masks
    '''
    _, H, W = masks.shape
    masks, flow = masks.to(device), flow.to(device)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y, x = y.to(device), x.to(device)
    y = y + flow[1]
    x = x + flow[0]
    y = (y / (H - 1)) * 2 - 1
    x = (x / (W - 1)) * 2 - 1
    grid = torch.stack((x, y)).permute(1, 2, 0).unsqueeze(0)
    masks = masks.unsqueeze(0)
    warped_masks = torch_f.grid_sample(masks, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
    return (warped_masks > 0.5) * 1.0

@torch.no_grad()
def compute_ious(masks1, masks2):
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
def random_sampling(mask, n_points=5):
    '''
    args:
        mask: torch.Tensor, (H, W), binary mask (0 or 1)
        n_points: int, number of points to sample
    returns:
        point_coords: np.ndarray, (n_points, 2), coordinates of sampled points
    '''
    y, x = torch.where(mask)
    points = torch.stack((x, y), dim=1).cpu().numpy()
    n_points = min(n_points, points.shape[0])
    if n_points == 0:
        return None
    sampled_indices = np.random.choice(points.shape[0], n_points, replace=False)
    return points[sampled_indices]

def segment_frame(frame, mask, color, alpha):
    '''
    args:
        frame: np.ndarray, (H, W, 3), frame
        mask: np.ndarray, (H, W), binary mask
        color: np.ndarray, (3,), color for overlaying mask
        alpha: float, alpha value for overlaying mask
    returns:
        segmented_frame: np.ndarray, (H, W, 3), segmented frame
    '''
    segmented_frame = frame.copy()
    coords = np.where(mask)
    segmented_frame[coords] = alpha * color + (1 - alpha) * segmented_frame[coords]
    edge = cv2.Canny(mask, 0, 1)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1)
    coords = np.where(edge)
    segmented_frame[coords] = color
    return segmented_frame.astype(np.uint8)

def segment_frame_AnnotatedImage(frame, mask, color):
    '''
    args:
        frame: np.ndarray, (H, W, 3), frame
        mask: np.ndarray, (H, W), binary mask
        color: np.ndarray, (3,), color for overlaying mask
        alpha: float, alpha value for overlaying mask
    returns:
        segmented_frame: List(np.ndarray, [np.ndarray, str]), segmented frame
    '''
    segmented_frame = frame.copy()
    annotated_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    coords = np.where(mask)
    annotated_mask[coords] = 1.0
    edge = cv2.Canny(mask, 0, 1)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1)
    coords = np.where(edge)
    annotated_mask[coords] = 1.0
    return [segmented_frame.astype(np.uint8), [[annotated_mask, color]]]

def compute_iou_numpy(mask1, mask2):
    '''
    args:
        mask1: np.ndarray, (H, W), binary mask
        mask2: np.ndarray, (H, W), binary mask
    returns:
        iou: float, intersection over union
    '''
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / (union + 1e-6)
    return iou

def get_pallete(num_cls=32):
    '''
    args:
        num_cls: int, number of classes
    returns:
        pallete: np.ndarray, (num_cls, 3), color pallete
    '''
    if num_cls > 999:
        num_cls = 999
    pallete_dict = {}
    hex_pallete_dict = {}
    for cls_idx in range(num_cls):
        color = np.random.randint(0, 256, 3)
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        pallete_dict[f'{cls_idx:03d}'] = color
        hex_pallete_dict[f'{cls_idx:03d}'] = hex_color
    return pallete_dict, hex_pallete_dict