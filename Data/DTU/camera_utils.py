"""Camera utility functions for DTU dataset.

This module provides clean, well-documented functions for camera-related operations
including projection matrix decomposition and coordinate transformations.
"""

import numpy as np
import torch
from typing import Tuple
from scipy.linalg import rq


def decompose_projection_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose projection matrix P = K[R|t] into intrinsics K, rotation R, and translation t.
    
    Args:
        P: 3x4 projection matrix
        
    Returns:
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        t: 3D translation vector (world to camera)
        
    Note:
        Uses RQ decomposition to ensure K has positive diagonal elements.
    """
    if P.shape != (3, 4):
        raise ValueError(f"Expected 3x4 projection matrix, got shape {P.shape}")
    
    # RQ decomposition of left 3x3 submatrix
    M = P[:, :3]
    K, R = rq(M)
    
    # Ensure positive diagonal elements in K
    # This is done by multiplying by appropriate sign matrix T
    diagonal_signs = np.sign(np.diag(K))
    T = np.diag(diagonal_signs)
    K = K @ T
    R = T @ R
    
    # Compute translation: t = K^(-1) * P[:, 3]
    t = np.linalg.solve(K, P[:, 3])
    
    # Normalize K so that K[2,2] = 1
    K = K / K[2, 2]
    
    return K, R, t


def numpy_to_torch_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert camera parameters from numpy arrays to torch tensors.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        R: Rotation matrix (3x3) 
        t: Translation vector (3,)
        
    Returns:
        Tuple of (K_torch, R_torch, t_torch) as float tensors
    """
    return (
        torch.from_numpy(K.copy()).float(),
        torch.from_numpy(R.copy()).float(),
        torch.from_numpy(t.copy()).float()
    )


def camera_to_world_pose(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world pose from world-to-camera parameters.
    
    Args:
        R: Rotation matrix from world to camera (3x3)
        t: Translation vector from world to camera (3,)
        
    Returns:
        R_world: Rotation matrix from camera to world (3x3)
        t_world: Translation vector (camera center in world coordinates) (3,)
    """
    R_world = R.T
    t_world = -R_world @ t
    return R_world, t_world


def project_points(points_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project 3D points to image coordinates.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        K: Camera intrinsic matrix (3, 3)
        R: Rotation matrix world-to-camera (3, 3)
        t: Translation vector world-to-camera (3,)
        
    Returns:
        points_2d: 2D image coordinates (N, 2)
    """
    # Transform to camera coordinates
    points_cam = (R @ points_3d.T + t.unsqueeze(-1)).T  # (N, 3)
    
    # Project to image plane
    points_proj = (K @ points_cam.T).T  # (N, 3)
    
    # Convert to 2D coordinates
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (N, 2)
    
    return points_2d


def depth_to_points_3d(depth: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Convert depth map to 3D points in world coordinates.
    
    Args:
        depth: Depth map (H, W)
        K: Camera intrinsic matrix (3, 3)  
        R: Rotation matrix world-to-camera (3, 3)
        t: Translation vector world-to-camera (3,)
        
    Returns:
        points_3d: 3D points in world coordinates (N, 3) where N is number of valid depth pixels
    """
    H, W = depth.shape
    device = depth.device
    
    # Create pixel grid
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    
    # Get valid depth pixels
    valid_mask = ~torch.isnan(depth) & (depth > 0)
    valid_depth = depth[valid_mask]  # (N,)
    valid_u = u[valid_mask].float()  # (N,)
    valid_v = v[valid_mask].float()  # (N,)
    
    # Unproject to camera coordinates
    K_inv = torch.inverse(K)
    pixel_coords = torch.stack([valid_u, valid_v, torch.ones_like(valid_u)], dim=1)  # (N, 3)
    cam_coords = (K_inv @ pixel_coords.T).T  # (N, 3)
    cam_coords = cam_coords * valid_depth.unsqueeze(-1)  # Scale by depth
    
    # Transform to world coordinates  
    R_inv = R.T  # Camera-to-world rotation
    t_world = -R_inv @ t  # Camera center in world coords
    world_coords = (R_inv @ cam_coords.T).T + t_world.unsqueeze(0)  # (N, 3)
    
    return world_coords


def compute_camera_frustum(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, 
                          image_width: int, image_height: int, 
                          near: float = 0.1, far: float = 10.0) -> torch.Tensor:
    """Compute camera frustum corners in world coordinates.
    
    Args:
        K: Camera intrinsic matrix (3, 3)
        R: Rotation matrix world-to-camera (3, 3)  
        t: Translation vector world-to-camera (3,)
        image_width: Image width in pixels
        image_height: Image height in pixels
        near: Near plane distance
        far: Far plane distance
        
    Returns:
        frustum_corners: 8 frustum corner points in world coordinates (8, 3)
                        Order: near plane (bottom-left, bottom-right, top-right, top-left),
                               far plane (bottom-left, bottom-right, top-right, top-left)
    """
    # Define image plane corners
    corners_2d = torch.tensor([
        [0, image_height],           # bottom-left
        [image_width, image_height], # bottom-right  
        [image_width, 0],            # top-right
        [0, 0]                       # top-left
    ], dtype=torch.float32, device=K.device)
    
    # Unproject to camera rays
    K_inv = torch.inverse(K)
    corners_homo = torch.cat([corners_2d, torch.ones(4, 1, device=K.device)], dim=1)  # (4, 3)
    ray_dirs = (K_inv @ corners_homo.T).T  # (4, 3)
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=1, keepdim=True)  # Normalize
    
    # Create near and far plane points
    near_points = ray_dirs * near  # (4, 3)
    far_points = ray_dirs * far    # (4, 3)
    
    # Stack near and far points
    frustum_cam = torch.cat([near_points, far_points], dim=0)  # (8, 3)
    
    # Transform to world coordinates
    R_inv = R.T
    t_world = -R_inv @ t
    frustum_world = (R_inv @ frustum_cam.T).T + t_world.unsqueeze(0)  # (8, 3)
    
    return frustum_world


def normalize_intrinsics(K: np.ndarray) -> np.ndarray:
    """Normalize camera intrinsic matrix so that K[2,2] = 1.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        
    Returns:
        K_normalized: Normalized intrinsic matrix
    """
    K_norm = K.copy()
    K_norm = K_norm / K_norm[2, 2]
    return K_norm


def scale_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scale camera intrinsic matrix for different image resolution.
    
    Args:
        K: Original camera intrinsic matrix (3x3)
        scale_x: Horizontal scaling factor
        scale_y: Vertical scaling factor
        
    Returns:
        K_scaled: Scaled intrinsic matrix
    """
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy  
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled