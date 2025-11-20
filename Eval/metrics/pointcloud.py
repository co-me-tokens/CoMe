import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import pypose as pp 
import typing as T

@dataclass
class PointcloudEvaluationResult:
    accuracy: float
    completeness: float
    chamfer_distance: float

def unproject_depth_to_pointcloud(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Unprojects a batch of depth maps to 3D point clouds using camera intrinsics.

    This function takes a depth map, which provides the Z-coordinate for each pixel,
    and the camera's intrinsic matrix to compute the corresponding X and Y coordinates
    for each point, effectively creating a 3D point cloud.

    Args:
        depth (torch.Tensor): A batch of depth maps with shape (B, 1, H, W), where
                              B is the batch size, H is the height, and W is the width.
        intrinsics (torch.Tensor): A batch of camera intrinsic matrices with shape (B, 3, 3).
                                   The matrix is expected to be in the format:
                                   [[fx, 0,  cx],
                                    [0,  fy, cy],
                                    [0,  0,  1]]

    Returns:
        torch.Tensor: A batch of 3D point clouds with shape (B, N, 3), where N = H * W.
                      Each point is represented by its (X, Y, Z) coordinates.
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    
    # Extract intrinsic parameters from the intrinsic matrices
    fx = intrinsics[:, 0, 0]  # Focal length in x direction (B,)
    fy = intrinsics[:, 1, 1]  # Focal length in y direction (B,)
    cx = intrinsics[:, 0, 2]  # Principal point x coordinate (B,)
    cy = intrinsics[:, 1, 2]  # Principal point y coordinate (B,)
    
    # Create pixel coordinate grids
    # u represents horizontal pixel coordinates (columns, 0 to W-1)
    # v represents vertical pixel coordinates (rows, 0 to H-1)
    u_coords = torch.arange(W, dtype=dtype, device=device)
    v_coords = torch.arange(H, dtype=dtype, device=device)
    
    # Create 2D grids of pixel coordinates
    # u_grid[i,j] = j (column index), v_grid[i,j] = i (row index)
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')
    
    # Expand grids to match batch dimension
    u_grid = u_grid.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, W)
    v_grid = v_grid.unsqueeze(0).expand(B, -1, -1)  # Shape: (B, H, W)
    
    # Remove channel dimension from depth to get Z coordinates
    z = depth.squeeze(1)  # Shape: (B, H, W)
    
    # Reshape intrinsic parameters for element-wise operations
    fx = fx.view(B, 1, 1)
    fy = fy.view(B, 1, 1)
    cx = cx.view(B, 1, 1)
    cy = cy.view(B, 1, 1)
    
    # Apply the unprojection formula to compute 3D coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    x = (u_grid - cx) * z / fx  # Shape: (B, H, W)
    y = (v_grid - cy) * z / fy  # Shape: (B, H, W)
    
    # Stack the coordinates to form 3D points
    pointcloud = torch.stack([x, y, z], dim=-1)  # Shape: (B, H, W, 3)
    
    # Reshape to (B, N, 3) where N = H * W
    pointcloud = pointcloud.reshape(B, H * W, 3)
    
    return pointcloud

def umeyama_alignment(
    src: torch.Tensor,
    dst: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    allow_reflection: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute similarity transform (R, t, s) aligning src -> dst using Umeyama (1991).
    Args:
        src: (N, D) source points
        dst: (N, D) destination points
        mask: (N,) optional boolean mask selecting valid correspondences
        allow_reflection: if True, allow det(R) to be -1; otherwise enforce proper rotation
    Returns:
        R: (D, D) rotation (orthogonal) matrix
        t: (D,) translation vector
        s: () scalar scale factor as a 0-D tensor
    """
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError("src and dst must be of shape (N, D)")
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        src = src[mask]
        dst = dst[mask]

    N, D = src.shape
    if N < 2:
        raise ValueError("Need at least 2 points")
    # For a stable SVD-based estimate of rotation in 3D, N>=3 is recommended
    if D == 3 and N < 3:
        raise ValueError("Need at least 3 points for 3D")

    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Variance of source (mean squared norm)
    var_src = torch.mean(torch.sum(src_c * src_c, dim=1))
    if var_src <= 0:
        raise ValueError("Degenerate source configuration (zero variance)")

    # Cross-covariance (normalized by N)
    Sigma = (src_c.T @ dst_c) / N  # D x D

    # SVD
    U, S, Vh = torch.linalg.svd(Sigma)  # Sigma = U @ diag(S) @ Vh
    V = Vh.T

    # Construct D matrix to handle reflection if needed
    det_uv = torch.det(V @ U.T)
    D_mat = torch.eye(D, dtype=src.dtype, device=src.device)
    if not allow_reflection:
        if det_uv < 0:
            D_mat[-1, -1] = -1.0
    else:
        # allow reflection: do not force D's last entry negative; keep identity
        pass

    # Rotation
    R = V @ D_mat @ U.T

    # Scale
    # s = trace(diag(S) @ D) / var_src
    s = (S * torch.diag(D_mat)).sum() / var_src

    t = dst_mean - s * (R @ src_mean)

    return R, t, s

def compute_chamfer_distance_torch3d(points1: torch.Tensor, points2: torch.Tensor, max_points: int, max_dist: float = 10000.) -> tuple[float, float, float]: 
    """
    Args: 
        points1: (N, 3) 
        points2: (M, 3) 
        max_points: maximum number of points to use 
    Returns: 
        accurarcy: float 
        completeness: float
        chamfer_distance: float
    """
    if points1.shape[0] > max_points:
        torch.manual_seed(33)
        indices = torch.randperm(points1.size(0))[:max_points]
        points1 = points1[indices]

    if points2.shape[0] > max_points:
        torch.manual_seed(33)
        indices = torch.randperm(points2.size(0))[:max_points]
        points2 = points2[indices]

    points1 = points1.unsqueeze(0)
    points2 = points2.unsqueeze(0)
    # Compute accuracy and completeness (directional chamfer distances)
    # breakpoint()
    (accuracy, completeness), _ = chamfer_distance(
        points1, points2, 
        norm=2, 
        point_reduction=None,  # Return per-point distances
        batch_reduction=None,
        single_directional=False
    )
    
    # accuracy: distance from prediction to ground truth (N, P1)
    # completeness: distance from ground truth to prediction (N, P2)
    accuracy = torch.clamp(torch.sqrt(accuracy), max=max_dist)
    completeness = torch.clamp(torch.sqrt(completeness), max=max_dist)
    accuracy_mean = accuracy.mean().item()
    completeness_mean = completeness.mean().item()
    
    # Chamfer distance is the average of accuracy and completeness
    chamfer_dist = (accuracy_mean + completeness_mean) / 2
    return accuracy_mean, completeness_mean, chamfer_dist

def compute_chamfer_distance_o3d(points_pred, points_gt, max_chamfer_points=100000, max_dist=10000.):
    MAX_POINTS = max_chamfer_points
    if points_pred.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(points_pred.shape[0], MAX_POINTS, replace=False)
        points_pred = points_pred[indices]

    if points_gt.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(points_gt.shape[0], MAX_POINTS, replace=False)
        points_gt = points_gt[indices]

    # Convert numpy point clouds to open3d point cloud objects
    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)

    # Downsample point clouds to accelerate computation
    # voxel_size = 0.05  # 5cm voxel size
    # pcd_pred = pcd_pred.voxel_down_sample(voxel_size)
    # pcd_gt = pcd_gt.voxel_down_sample(voxel_size)

    # Compute distances from predicted point cloud to GT point cloud
    distances1 = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    # Compute distances from GT point cloud to predicted point cloud
    distances2 = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))

    # Apply distance clipping
    accuracy = np.mean(np.clip(distances1, 0, max_dist)).item()
    completness = np.mean(np.clip(distances2, 0, max_dist)).item()

    # Chamfer Distance is the sum of mean distances in both directions
    chamfer_dist = (accuracy + completness) / 2

    return accuracy, completness, chamfer_dist

def apply_similarity_transform(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor, s: float | torch.Tensor) -> torch.Tensor:
    """Apply similarity transformation: s * R * points + t"""
    return s * (points @ R.T) + t.unsqueeze(0)



def filter_valid_correspondences(pred_points: torch.Tensor, gt_points: torch.Tensor,
                                max_distance: float = 0.1) -> torch.Tensor:
    """
    Filter valid correspondences based on distance threshold
    Args:
        pred_points: (N, 3) predicted points
        gt_points: (N, 3) ground truth points (same size as pred)
        max_distance: maximum distance threshold for valid correspondence
    Returns:
        mask: (N,) boolean mask for valid correspondences
    """
    distances = torch.norm(pred_points - gt_points, dim=1)
    return distances < max_distance


def evaluation_pipeline(pred_pointclouds: torch.Tensor, 
                       gt_depths: torch.Tensor,
                       gt_pointclouds: torch.Tensor,
                       intrinsics: torch.Tensor,
                       poses: pp.LieTensor,
                       mask: Optional[torch.Tensor] = None) -> PointcloudEvaluationResult:
    """
    Complete pointcloud evaluation pipeline
    Args:
        pred_pointclouds: (S, 3, H, W) predicted point clouds per frame
        gt_depths: (S, 1, H, W) ground truth depth maps
        gt_pointclouds: (M, 3) ground truth point cloud
        intrinsics: (S, 3, 3) camera intrinsics per frame
        poses: (S, 7) world to cam pp SE3 Lietensor
        mask: (S, H, W) Optional mask for evaluation region
    Returns:
        PointcloudEvaluationResult with evaluation metrics
    """
    device = gt_depths.device
    S = gt_depths.shape[0]
    # gt_depths = ppgt_depths[mask] 
    # 1. Unproject GT depth maps to point clouds
    gt_pointclouds_from_depth = unproject_depth_to_pointcloud(gt_depths, intrinsics)  # (S, N, 3)
    N = gt_pointclouds_from_depth.shape[1]
    poses = T.cast(pp.LieTensor, poses.unsqueeze(1))  
    gt_pointclouds_from_depth_world = pp.Act(poses.Inv().expand(-1, N, -1).reshape(-1, 7), gt_pointclouds_from_depth.reshape(-1, 3)).reshape(-1, 3)
    pred_pointclouds_flatten = pred_pointclouds.permute(0, 2, 3, 1).reshape(-1, 3)  # (S, N, 3) 
    if mask is not None: 
        pred_pointclouds_flatten = pred_pointclouds_flatten[mask.reshape(-1)]
        gt_pointclouds_from_depth_world = gt_pointclouds_from_depth_world[mask.reshape(-1)]
    max_alignment_points = pred_pointclouds_flatten.shape[0]
    n = len(pred_pointclouds_flatten)
    assert n == len(gt_pointclouds_from_depth_world), "pred and gt must have the same length to share a permutation."
    if n > max_alignment_points:
        torch.manual_seed(42)
        indices = torch.randperm(n)[:max_alignment_points]
        pred_sample = pred_pointclouds_flatten[indices]
        gt_sample = gt_pointclouds_from_depth_world[indices]
    else:
        pred_sample = pred_pointclouds_flatten
        gt_sample = gt_pointclouds_from_depth_world
    
    R, t, s = umeyama_alignment(pred_sample, gt_sample, None) 

    pred_aligned = apply_similarity_transform(pred_pointclouds_flatten, R, t, s)

    max_chamfer_points = 100000 
    accuracy, completeness, chamfer_dist = compute_chamfer_distance_o3d(pred_aligned.cpu().numpy(), gt_pointclouds.cpu().numpy(), max_chamfer_points)
    # compute_chamfer_distance_torch3d(pred_aligned, gt_pointclouds, max_chamfer_points)
    return PointcloudEvaluationResult(
        accuracy=accuracy,
        completeness=completeness,
        chamfer_distance=chamfer_dist,
    )

    