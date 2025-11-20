"""Visualization utilities for DTU dataset.

This module provides clean, well-documented visualization functions for DTU dataset
using various backends (matplotlib, rerun, open3d) with proper error handling.
"""

import os
import numpy as np
import torch
from typing import Optional, List, Tuple, Union, Any
import cv2

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    rr = None
    RERUN_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False


def visualize_dataset_rerun(dataset, app_name: str = "DTU Dataset Visualization", 
                          max_cameras: Optional[int] = None) -> None:
    """Visualize DTU dataset using Rerun.
    
    Args:
        dataset: DTUDataset instance
        app_name: Application name for Rerun
        max_cameras: Maximum number of cameras to visualize (None for all)
    """
    if not RERUN_AVAILABLE:
        raise ImportError("rerun-sdk not installed. Run: pip install rerun-sdk")
    
    # Initialize Rerun
    rr.init(app_name, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)
    
    # Determine number of cameras to show
    n_cameras = len(dataset)
    if max_cameras is not None:
        n_cameras = min(n_cameras, max_cameras)
    
    print(f"Visualizing {n_cameras} cameras...")
    
    # Log cameras and images
    for idx in range(n_cameras):
        sample = dataset[idx]
        
        # Load image
        image = cv2.imread(sample["image_path"])
        if image is None:
            print(f"Warning: Could not load image {sample['image_path']}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Camera pose: convert from world-to-camera to camera-to-world
        R_world_to_cam = sample["R"]  # World to camera rotation
        t_world_to_cam = sample["t"]  # World to camera translation
        
        # Camera-to-world transformation
        R_cam_to_world = R_world_to_cam.T
        camera_center = -R_cam_to_world @ t_world_to_cam
        
        # Log camera
        camera_path = f"cameras/camera_{sample['view_id']:03d}"
        rr.log(
            camera_path,
            rr.Pinhole(
                image_from_camera=sample["K"].numpy(),
                resolution=(image_rgb.shape[1], image_rgb.shape[0])
            ),
            rr.Transform3D(
                translation=camera_center.numpy(),
                mat3x3=R_cam_to_world.numpy()
            ),
            rr.Image(image_rgb),
        )
    
    # Log point cloud if available
    try:
        vertices, colors = dataset.get_point_cloud()
        rr.log(
            "point_cloud",
            rr.Points3D(
                positions=vertices.numpy(),
                colors=colors.numpy()
            )
        )
        print(f"Point cloud logged with {len(vertices)} points")
    except Exception as e:
        print(f"Warning: Could not load point cloud: {e}")
    
    print("✅ Visualization complete. Check Rerun viewer.")


def visualize_cameras_matplotlib(dataset, num_cameras: Optional[int] = None, 
                               show_point_cloud: bool = True,
                               camera_scale: float = 0.1) -> None:
    """Visualize DTU cameras using matplotlib.
    
    Args:
        dataset: DTUDataset instance
        num_cameras: Number of cameras to show (None for all)
        show_point_cloud: Whether to show point cloud
        camera_scale: Scale factor for camera visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine cameras to show
    n_show = len(dataset) if num_cameras is None else min(num_cameras, len(dataset))
    
    camera_centers = []
    camera_directions = []
    
    # Plot cameras
    for idx in range(n_show):
        sample = dataset[idx]
        
        # Get camera center and orientation
        R = sample["R"]  # World to camera
        t = sample["t"]
        
        # Camera center in world coordinates
        R_inv = R.T
        center = -R_inv @ t
        camera_centers.append(center.numpy())
        
        # Camera forward direction (negative Z in camera frame)
        forward = R_inv @ np.array([0, 0, -1])  # Camera looks along -Z
        camera_directions.append(forward)
        
        # Plot camera center
        ax.scatter(*center.numpy(), c='red', s=50, alpha=0.7)
        
        # Plot camera direction
        end_point = center.numpy() + forward * camera_scale
        ax.plot([center[0], end_point[0]], 
               [center[1], end_point[1]], 
               [center[2], end_point[2]], 'r-', alpha=0.7)
    
    camera_centers = np.array(camera_centers)
    
    # Plot point cloud (subsampled for performance)
    if show_point_cloud:
        try:
            vertices, colors = dataset.get_point_cloud()
            vertices_np = vertices.numpy()
            
            # Subsample if too many points
            if len(vertices_np) > 10000:
                indices = np.random.choice(len(vertices_np), 10000, replace=False)
                vertices_np = vertices_np[indices]
                colors_np = colors.numpy()[indices] / 255.0  # Normalize to [0,1]
            else:
                colors_np = colors.numpy() / 255.0
            
            ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                      c=colors_np, s=1, alpha=0.5)
        except Exception as e:
            print(f"Warning: Could not load point cloud: {e}")
    
    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'DTU Dataset - Scan {dataset.scan_id} ({n_show} cameras)')
    
    # Set equal aspect ratio
    if len(camera_centers) > 0:
        # Use camera centers to set reasonable bounds
        center_bounds = np.array([
            [camera_centers[:, 0].min(), camera_centers[:, 0].max()],
            [camera_centers[:, 1].min(), camera_centers[:, 1].max()],
            [camera_centers[:, 2].min(), camera_centers[:, 2].max()]
        ])
        
        # Expand bounds slightly
        center_range = np.max(center_bounds[:, 1] - center_bounds[:, 0])
        center_center = np.mean(center_bounds, axis=1)
        
        bound_size = center_range * 0.6
        ax.set_xlim(center_center[0] - bound_size, center_center[0] + bound_size)
        ax.set_ylim(center_center[1] - bound_size, center_center[1] + bound_size)
        ax.set_zlim(center_center[2] - bound_size, center_center[2] + bound_size)
    
    plt.tight_layout()
    plt.show()


def visualize_point_cloud_open3d(dataset, filter_bounds: Optional[Tuple[float, float]] = None) -> None:
    """Visualize DTU point cloud using Open3D.
    
    Args:
        dataset: DTUDataset instance
        filter_bounds: Optional (min_distance, max_distance) from origin to filter points
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d not installed. Run: pip install open3d")
    
    try:
        vertices, colors = dataset.get_point_cloud()
        vertices_np = vertices.numpy()
        colors_np = colors.numpy()
        
        # Apply distance filtering if requested
        if filter_bounds is not None:
            distances = np.linalg.norm(vertices_np, axis=1)
            mask = (distances >= filter_bounds[0]) & (distances <= filter_bounds[1])
            vertices_np = vertices_np[mask]
            colors_np = colors_np[mask]
            print(f"Filtered to {len(vertices_np)} points (distance range: {filter_bounds})")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np / 255.0)  # Normalize to [0,1]
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], 
                                        window_name=f"DTU Scan {dataset.scan_id} Point Cloud",
                                        width=1200, height=800)
        
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")


def plot_depth_map(depth: torch.Tensor, title: str = "Depth Map", 
                  save_path: Optional[str] = None, colormap: str = 'viridis') -> None:
    """Plot depth map using matplotlib.
    
    Args:
        depth: Depth map tensor (H, W)
        title: Plot title
        save_path: Optional path to save plot
        colormap: Matplotlib colormap name
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")
    
    depth_np = depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth
    
    plt.figure(figsize=(10, 6))
    
    # Handle NaN values for visualization
    valid_mask = ~np.isnan(depth_np)
    if np.any(valid_mask):
        vmin, vmax = np.nanmin(depth_np), np.nanmax(depth_np)
        plt.imshow(depth_np, cmap=colormap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Depth')
        plt.title(f"{title} (Range: {vmin:.2f} - {vmax:.2f})")
    else:
        plt.imshow(depth_np, cmap=colormap)
        plt.colorbar(label='Depth')
        plt.title(f"{title} (All NaN)")
    
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Depth map saved to: {save_path}")
    
    plt.show()


def create_camera_trajectory_plot(dataset, output_path: Optional[str] = None) -> None:
    """Create a plot showing camera trajectory.
    
    Args:
        dataset: DTUDataset instance  
        output_path: Optional path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")
    
    camera_centers = []
    view_ids = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        R = sample["R"]
        t = sample["t"] 
        
        # Camera center in world coordinates
        center = -R.T @ t
        camera_centers.append(center.numpy())
        view_ids.append(sample["view_id"])
    
    camera_centers = np.array(camera_centers)
    
    # Create trajectory plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top view (X-Y plane)
    ax1.scatter(camera_centers[:, 0], camera_centers[:, 1], c=range(len(camera_centers)), 
               cmap='viridis', s=50)
    ax1.plot(camera_centers[:, 0], camera_centers[:, 1], 'k-', alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Camera Trajectory (Top View) - Scan {dataset.scan_id}')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Side view (X-Z plane)
    ax2.scatter(camera_centers[:, 0], camera_centers[:, 2], c=range(len(camera_centers)), 
               cmap='viridis', s=50)
    ax2.plot(camera_centers[:, 0], camera_centers[:, 2], 'k-', alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title(f'Camera Trajectory (Side View) - Scan {dataset.scan_id}')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=0, vmax=len(camera_centers)-1))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], label='Camera Index')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {output_path}")
    
    plt.show()


def compare_depth_maps(depth1: torch.Tensor, depth2: torch.Tensor, 
                      labels: List[str] = ["Depth 1", "Depth 2"],
                      save_path: Optional[str] = None) -> None:
    """Compare two depth maps side by side.
    
    Args:
        depth1: First depth map
        depth2: Second depth map  
        labels: Labels for the depth maps
        save_path: Optional path to save comparison
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")
    
    depth1_np = depth1.cpu().numpy() if isinstance(depth1, torch.Tensor) else depth1
    depth2_np = depth2.cpu().numpy() if isinstance(depth2, torch.Tensor) else depth2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot first depth map
    im1 = axes[0].imshow(depth1_np, cmap='viridis')
    axes[0].set_title(labels[0])
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot second depth map  
    im2 = axes[1].imshow(depth2_np, cmap='viridis')
    axes[1].set_title(labels[1])
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    valid_mask = ~(np.isnan(depth1_np) | np.isnan(depth2_np))
    diff = np.full_like(depth1_np, np.nan)
    diff[valid_mask] = np.abs(depth1_np[valid_mask] - depth2_np[valid_mask])
    
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Depth comparison saved to: {save_path}")
    
    plt.show()


def export_camera_poses(dataset, output_path: str, format: str = "json") -> None:
    """Export camera poses to file.
    
    Args:
        dataset: DTUDataset instance
        output_path: Output file path
        format: Export format ("json" or "txt")
    """
    poses_data = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        
        pose_dict = {
            "view_id": int(sample["view_id"]),
            "image_path": sample["image_path"],
            "K": sample["K"].numpy().tolist(),
            "R": sample["R"].numpy().tolist(), 
            "t": sample["t"].numpy().tolist()
        }
        poses_data.append(pose_dict)
    
    if format == "json":
        import json
        with open(output_path, 'w') as f:
            json.dump(poses_data, f, indent=2)
    elif format == "txt":
        with open(output_path, 'w') as f:
            f.write(f"# DTU Scan {dataset.scan_id} Camera Poses\n")
            f.write("# view_id K[3x3] R[3x3] t[3x1]\n")
            for pose in poses_data:
                f.write(f"{pose['view_id']} ")
                
                # Write K matrix (row-major)
                K = np.array(pose['K'])
                f.write(" ".join(f"{x:.6f}" for x in K.flatten()))
                f.write(" ")
                
                # Write R matrix (row-major)  
                R = np.array(pose['R'])
                f.write(" ".join(f"{x:.6f}" for x in R.flatten()))
                f.write(" ")
                
                # Write t vector
                t = np.array(pose['t'])
                f.write(" ".join(f"{x:.6f}" for x in t))
                f.write("\n")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Camera poses exported to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("DTU Visualization Utilities")
    print("Available backends:")
    print(f"  matplotlib: {'✅' if MATPLOTLIB_AVAILABLE else '❌'}")
    print(f"  rerun: {'✅' if RERUN_AVAILABLE else '❌'}")  
    print(f"  open3d: {'✅' if OPEN3D_AVAILABLE else '❌'}")
    
    # Example with mock data
    if MATPLOTLIB_AVAILABLE:
        print("\nTesting matplotlib depth visualization...")
        depth = torch.randn(100, 150) * 5 + 10  # Mock depth data
        depth[depth < 0] = float('nan')  # Some invalid depths
        plot_depth_map(depth, "Test Depth Map")