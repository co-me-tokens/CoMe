"""VGGT geometry utilities.

Functions ported from the official VGGT repository
(https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py)
with the distortion dependency removed.
"""

import torch
import numpy as np


def closed_form_inverse_se3(
    se3: np.ndarray | torch.Tensor,
    R: np.ndarray | torch.Tensor | None = None,
    T: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Compute the inverse of each SE3 matrix in a batch.

    Args:
        se3: Nx4x4 or Nx3x4 array/tensor of SE3 matrices.
        R:   Optional Nx3x3 rotation matrices (extracted from *se3* when None).
        T:   Optional Nx3x1 translation vectors (extracted from *se3* when None).

    Returns:
        Inverted SE3 matrices (Nx4x4) with the same type and device as *se3*.
    """
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be Nx4x4 or Nx3x4, got {se3.shape}")

    is_numpy = isinstance(se3, np.ndarray)

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_t = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_t, T)
        inv = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_t = R.transpose(1, 2)
        top_right = -torch.bmm(R_t, T)
        inv = torch.eye(4, device=se3.device, dtype=R.dtype).unsqueeze(0).repeat(len(R), 1, 1)

    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = top_right
    return inv


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Convert a single depth map to camera-local 3D coordinates.

    Args:
        depth_map: (H, W) depth values.
        intrinsic: (3, 3) camera intrinsic matrix.

    Returns:
        (H, W, 3) camera-frame 3D points.
    """
    assert intrinsic.shape == (3, 3)
    H, W = depth_map.shape

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single depth map to world coordinates.

    Args:
        depth_map: (H, W) depth values.
        extrinsic: (3, 4) world-to-camera extrinsic matrix (OpenCV convention).
        intrinsic: (3, 3) camera intrinsic matrix.

    Returns:
        (world_coords, cam_coords, point_mask) -- each (H, W, 3), (H, W, 3), (H, W) bool.
    """
    if depth_map is None:
        return None, None, None  # type: ignore[return-value]

    point_mask = depth_map > eps
    cam_coords = depth_to_cam_coords_points(depth_map, intrinsic)

    cam_to_world = closed_form_inverse_se3(extrinsic[None])[0]
    R_c2w = cam_to_world[:3, :3]
    t_c2w = cam_to_world[:3, 3]

    world_coords = np.dot(cam_coords, R_c2w.T) + t_c2w
    return world_coords, cam_coords, point_mask


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray | torch.Tensor,
    extrinsics_cam: np.ndarray | torch.Tensor,
    intrinsics_cam: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map:      (S, H, W, 1) or (S, H, W) depth maps.
        extrinsics_cam: (S, 3, 4) world-to-camera extrinsic matrices.
        intrinsics_cam: (S, 3, 3) camera intrinsic matrices.

    Returns:
        (S, H, W, 3) world-frame 3D point maps.
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for i in range(depth_map.shape[0]):
        world_pts, _, _ = depth_to_world_coords_points(
            depth_map[i].squeeze(-1), extrinsics_cam[i], intrinsics_cam[i]
        )
        world_points_list.append(world_pts)
    return np.stack(world_points_list, axis=0)
