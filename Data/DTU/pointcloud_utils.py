"""Point cloud utilities for DTU dataset.

This module provides clean utilities for loading, processing, and manipulating
DTU point clouds with better error handling and documentation.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional, Union

try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PlyData = None
    PlyElement = None
    PLYFILE_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False


def load_ply_vertices(ply_path: str, backend: str = "plyfile") -> Tuple[np.ndarray, np.ndarray]:
    """Load point cloud vertices and colors from PLY file.
    
    Args:
        ply_path: Path to PLY file
        backend: Backend to use ("plyfile" or "open3d")
        
    Returns:
        vertices: Point coordinates as float32 array (N, 3) 
        colors: RGB colors as uint8 array (N, 3)
        
    Raises:
        ImportError: If required backend is not installed
        FileNotFoundError: If PLY file doesn't exist
        ValueError: If PLY file format is invalid
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    if backend == "plyfile":
        return _load_ply_with_plyfile(ply_path)
    elif backend == "open3d":
        return _load_ply_with_open3d(ply_path) 
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'plyfile' or 'open3d'")


def _load_ply_with_plyfile(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load PLY file using plyfile library."""
    if not PLYFILE_AVAILABLE:
        raise ImportError("plyfile not installed. Run: pip install plyfile")
    
    try:
        ply_data = PlyData.read(ply_path)
        vertices = ply_data["vertex"]
        
        # Extract coordinates
        coords = np.column_stack([
            vertices["x"].astype(np.float32),
            vertices["y"].astype(np.float32), 
            vertices["z"].astype(np.float32)
        ])
        
        # Extract colors (handle missing color case)
        if "red" in vertices.dtype.names:
            colors = np.column_stack([
                vertices["red"].astype(np.uint8),
                vertices["green"].astype(np.uint8),
                vertices["blue"].astype(np.uint8)
            ])
        else:
            # Default to white if no colors
            colors = np.full((len(vertices), 3), 255, dtype=np.uint8)
        
        return coords, colors
        
    except Exception as e:
        raise ValueError(f"Failed to load PLY file {ply_path}: {e}")


def _load_ply_with_open3d(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load PLY file using Open3D library."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d not installed. Run: pip install open3d")
    
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if len(pcd.points) == 0:
            raise ValueError("No points found in PLY file")
        
        # Extract coordinates
        coords = np.asarray(pcd.points, dtype=np.float32)
        
        # Extract colors
        if len(pcd.colors) > 0:
            # Open3D colors are in [0,1] range, convert to [0,255]
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            # Default to white if no colors
            colors = np.full((len(coords), 3), 255, dtype=np.uint8)
        
        return coords, colors
        
    except Exception as e:
        raise ValueError(f"Failed to load PLY file {ply_path}: {e}")


def save_ply_vertices(ply_path: str, vertices: np.ndarray, colors: Optional[np.ndarray] = None,
                     backend: str = "plyfile") -> None:
    """Save point cloud vertices and colors to PLY file.
    
    Args:
        ply_path: Output PLY file path
        vertices: Point coordinates (N, 3)
        colors: RGB colors (N, 3), optional
        backend: Backend to use ("plyfile" or "open3d")
    """
    if backend == "plyfile":
        _save_ply_with_plyfile(ply_path, vertices, colors)
    elif backend == "open3d":
        _save_ply_with_open3d(ply_path, vertices, colors)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _save_ply_with_plyfile(ply_path: str, vertices: np.ndarray, colors: Optional[np.ndarray]) -> None:
    """Save PLY file using plyfile library."""
    if not PLYFILE_AVAILABLE:
        raise ImportError("plyfile not installed. Run: pip install plyfile")
    
    vertices = vertices.astype(np.float32)
    n_points = len(vertices)
    
    # Create vertex data
    if colors is not None:
        colors = colors.astype(np.uint8)
        vertex_data = np.empty(n_points, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex_data['x'] = vertices[:, 0]
        vertex_data['y'] = vertices[:, 1] 
        vertex_data['z'] = vertices[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]
    else:
        vertex_data = np.empty(n_points, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
        ])
        vertex_data['x'] = vertices[:, 0]
        vertex_data['y'] = vertices[:, 1]
        vertex_data['z'] = vertices[:, 2]
    
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element]).write(ply_path)


def _save_ply_with_open3d(ply_path: str, vertices: np.ndarray, colors: Optional[np.ndarray]) -> None:
    """Save PLY file using Open3D library."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d not installed. Run: pip install open3d")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    
    if colors is not None:
        # Convert from [0,255] to [0,1] range for Open3D
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    o3d.io.write_point_cloud(ply_path, pcd)


def filter_point_cloud(vertices: np.ndarray, colors: np.ndarray, 
                      bbox_min: Optional[np.ndarray] = None,
                      bbox_max: Optional[np.ndarray] = None,
                      max_distance_from_origin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Filter point cloud based on spatial constraints.
    
    Args:
        vertices: Point coordinates (N, 3)
        colors: Point colors (N, 3) 
        bbox_min: Minimum bounds for bounding box filter (3,)
        bbox_max: Maximum bounds for bounding box filter (3,)
        max_distance_from_origin: Maximum distance from origin
        
    Returns:
        filtered_vertices: Filtered point coordinates
        filtered_colors: Filtered point colors
    """
    mask = np.ones(len(vertices), dtype=bool)
    
    # Apply bounding box filter
    if bbox_min is not None:
        mask &= np.all(vertices >= bbox_min, axis=1)
    if bbox_max is not None:
        mask &= np.all(vertices <= bbox_max, axis=1)
    
    # Apply distance filter  
    if max_distance_from_origin is not None:
        distances = np.linalg.norm(vertices, axis=1)
        mask &= distances <= max_distance_from_origin
    
    return vertices[mask], colors[mask]


def subsample_point_cloud(vertices: np.ndarray, colors: np.ndarray, 
                         max_points: int, method: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
    """Subsample point cloud to reduce number of points.
    
    Args:
        vertices: Point coordinates (N, 3)
        colors: Point colors (N, 3)
        max_points: Maximum number of points to keep
        method: Subsampling method ("uniform" or "random")
        
    Returns:
        subsampled_vertices: Subsampled point coordinates
        subsampled_colors: Subsampled point colors
    """
    n_points = len(vertices)
    if n_points <= max_points:
        return vertices, colors
    
    if method == "uniform":
        # Uniform subsampling
        step = n_points // max_points
        indices = np.arange(0, n_points, step)[:max_points]
    elif method == "random":
        # Random subsampling
        indices = np.random.choice(n_points, size=max_points, replace=False)
    else:
        raise ValueError(f"Unknown subsampling method: {method}")
    
    return vertices[indices], colors[indices]


def compute_point_cloud_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of point cloud.
    
    Args:
        vertices: Point coordinates (N, 3)
        
    Returns:
        bbox_min: Minimum coordinates (3,)
        bbox_max: Maximum coordinates (3,)
    """
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def center_point_cloud(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center point cloud at origin.
    
    Args:
        vertices: Point coordinates (N, 3)
        
    Returns:
        centered_vertices: Centered point coordinates
        centroid: Original centroid that was subtracted (3,)
    """
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    return centered_vertices, centroid


def scale_point_cloud(vertices: np.ndarray, target_scale: float) -> Tuple[np.ndarray, float]:
    """Scale point cloud to fit within target scale.
    
    Args:
        vertices: Point coordinates (N, 3)
        target_scale: Target maximum extent
        
    Returns:
        scaled_vertices: Scaled point coordinates  
        scale_factor: Applied scale factor
    """
    bbox_min, bbox_max = compute_point_cloud_bounds(vertices)
    current_scale = np.max(bbox_max - bbox_min)
    scale_factor = target_scale / current_scale
    scaled_vertices = vertices * scale_factor
    return scaled_vertices, scale_factor


def numpy_to_torch_pointcloud(vertices: np.ndarray, colors: np.ndarray, 
                            device: Optional[Union[str, torch.device]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy point cloud to PyTorch tensors.
    
    Args:
        vertices: Point coordinates (N, 3)
        colors: Point colors (N, 3)
        device: Target device for tensors
        
    Returns:
        vertices_torch: Point coordinates as tensor
        colors_torch: Point colors as tensor
    """
    vertices_torch = torch.from_numpy(vertices.copy()).float()
    colors_torch = torch.from_numpy(colors.copy())
    
    if device is not None:
        vertices_torch = vertices_torch.to(device)
        colors_torch = colors_torch.to(device)
    
    return vertices_torch, colors_torch


def torch_to_numpy_pointcloud(vertices: torch.Tensor, colors: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PyTorch point cloud tensors to numpy arrays.
    
    Args:
        vertices: Point coordinates tensor (N, 3)  
        colors: Point colors tensor (N, 3)
        
    Returns:
        vertices_np: Point coordinates as numpy array
        colors_np: Point colors as numpy array
    """
    vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
    colors_np = colors.detach().cpu().numpy().astype(np.uint8)
    return vertices_np, colors_np


def get_point_cloud_statistics(vertices: np.ndarray) -> dict:
    """Compute statistics for point cloud.
    
    Args:
        vertices: Point coordinates (N, 3)
        
    Returns:
        Dictionary with point cloud statistics
    """
    n_points = len(vertices)
    centroid = np.mean(vertices, axis=0)
    bbox_min, bbox_max = compute_point_cloud_bounds(vertices)
    extent = bbox_max - bbox_min
    
    # Compute distances from centroid
    distances = np.linalg.norm(vertices - centroid, axis=1)
    
    return {
        "n_points": n_points,
        "centroid": centroid,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max, 
        "extent": extent,
        "max_extent": np.max(extent),
        "mean_distance_from_centroid": np.mean(distances),
        "std_distance_from_centroid": np.std(distances),
        "max_distance_from_centroid": np.max(distances)
    }


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
        
        if not os.path.exists(ply_file):
            print(f"PLY file not found: {ply_file}")
            sys.exit(1)
        
        print(f"Loading point cloud: {ply_file}")
        
        # Try different backends
        for backend in ["plyfile", "open3d"]:
            try:
                vertices, colors = load_ply_vertices(ply_file, backend=backend)
                print(f"✅ Successfully loaded with {backend}")
                print(f"   Points: {len(vertices)}")
                print(f"   Coordinates shape: {vertices.shape}")
                print(f"   Colors shape: {colors.shape}")
                
                # Compute statistics
                stats = get_point_cloud_statistics(vertices)
                print(f"   Centroid: {stats['centroid']}")
                print(f"   Extent: {stats['extent']}")
                print(f"   Max extent: {stats['max_extent']:.2f}")
                break
                
            except ImportError as e:
                print(f"⚠️ {backend} not available: {e}")
            except Exception as e:
                print(f"❌ Failed to load with {backend}: {e}")
    else:
        print("Usage: python pointcloud_utils.py <ply_file>")
        print("Tests point cloud loading functionality")