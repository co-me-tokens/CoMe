"""DTU Dataset implementation with improved readability and simplified structure."""

import os
import glob
import numpy as np
from typing import Optional, Dict, Tuple, Any
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


class DTUDataset(Dataset):
    """DTU Multi-View Stereo Dataset.
    
    This dataset loads DTU MVS images with camera parameters and point clouds.
    Simplified from the original fragmented implementation for better readability.
    """
    
    def __init__(
        self,
        root: str,
        scan_id: int,
        use_rectified: bool = True,
        lighting_filter: Optional[str] = "_3_",
        num_images: int = -1,
        point_cloud_method: str = "stl",
        transform=None,
        seed: Optional[int] = None,
    ):
        """Initialize DTU dataset.
        
        Args:
            root: Path to DTU dataset root directory
            scan_id: Scan ID (1-128)
            use_rectified: Use rectified images instead of cleaned ones
            lighting_filter: Filter images by lighting condition (e.g., "_3_")
            num_images: Limit number of images (-1 for all)
            point_cloud_method: Point cloud method ("stl" or "mvsnet")  
            transform: Image transforms to apply
            seed: Random seed for image sampling
        """
        self.root = root
        self.scan_id = scan_id
        self.use_rectified = use_rectified
        self.transform = transform
        
        # Build file paths
        self._setup_paths()
        
        # Load and filter images and calibrations
        self._load_data(lighting_filter, num_images, seed)
        
        # Load point cloud
        self._load_point_cloud(point_cloud_method)
        
        # Cache for camera poses
        self._pose_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    
    def _setup_paths(self):
        """Setup directory paths based on dataset structure."""
        self.mvs_root = self.root
        
        # Image directory
        image_subdir = "Rectified" if self.use_rectified else "Cleaned"
        self.image_dir = os.path.join(self.mvs_root, image_subdir, f"scan{self.scan_id}")
        
        # Calibration directory  
        self.calib_dir = os.path.join(self.mvs_root, "Calibration", "cal18")
        
        # Point cloud directory
        self.points_dir = os.path.join(self.mvs_root, "Points")
    
    def _load_data(self, lighting_filter: Optional[str], num_images: int, seed: Optional[int]):
        """Load and match image and calibration files."""
        # Find image files
        img_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))
        if lighting_filter:
            img_paths = [p for p in img_paths if lighting_filter in os.path.basename(p)]
        if not img_paths:
            raise RuntimeError(f"No images found with lighting filter '{lighting_filter}', {self.image_dir}")
        
        # Find calibration files
        calib_paths = sorted(glob.glob(os.path.join(self.calib_dir, "*.txt")))
        if not calib_paths:
            raise RuntimeError("No calibration files found")
        
        # Match by view ID
        img_dict = {self._parse_view_id(p): p for p in img_paths}
        calib_dict = {self._parse_view_id(p): p for p in calib_paths}
        
        # Find common view IDs
        common_ids = sorted(set(img_dict.keys()) & set(calib_dict.keys()))
        if not common_ids:
            raise RuntimeError("No matching view IDs between images and calibrations")
        
        # Sample subset if requested
        if 0 < num_images < len(common_ids):
            common_ids = common_ids[:num_images]
        
        # Store final lists
        self.view_ids = common_ids
        self.image_paths = [img_dict[vid] for vid in common_ids]
        self.calib_paths = [calib_dict[vid] for vid in common_ids]
    
    def _load_point_cloud(self, method: str):
        """Load point cloud data."""
        points_method_dir = os.path.join(self.points_dir, method)
        ply_pattern = os.path.join(points_method_dir, f"{method}{self.scan_id:03d}_total.ply")
        ply_files = glob.glob(ply_pattern)
        
        if not ply_files:
            raise RuntimeError(f"Point cloud file not found: {ply_pattern}")
        
        self.point_cloud_vertices, self.point_cloud_colors = self._load_ply_file(ply_files[0])
    
    def _parse_view_id(self, file_path: str) -> int:
        """Extract numeric view ID from filename."""
        import re
        basename = os.path.basename(file_path)
        match = re.search(r'(\d+)', basename)
        if not match:
            raise ValueError(f"Cannot extract view ID from {file_path}")
        return int(match.group(1))
    
    def _load_ply_file(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load PLY point cloud file."""
        if PlyData is None:
            raise ImportError("Please install plyfile: pip install plyfile")
        
        ply_data = PlyData.read(ply_path)
        vertices = ply_data["vertex"]
        
        # Extract coordinates
        coords = np.column_stack([
            vertices["x"], vertices["y"], vertices["z"]
        ]).astype(np.float32)
        
        # Extract colors  
        colors = np.column_stack([
            vertices["red"], vertices["green"], vertices["blue"]
        ]).astype(np.uint8)
        
        return coords, colors
    
    def _read_projection_matrix(self, calib_path: str) -> np.ndarray:
        """Read 3x4 projection matrix from calibration file."""
        import re
        values = []
        with open(calib_path, 'r') as f:
            content = f.read()
            # Extract all floating point numbers
            float_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
            matches = re.findall(float_pattern, content)
            values = [float(x) for x in matches]
        
        if len(values) < 12:
            raise ValueError(f"Expected at least 12 values for 3x4 matrix, got {len(values)}")
        
        return np.array(values[:12]).reshape(3, 4)
    
    def _decompose_projection_matrix(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose projection matrix P = K[R|t] into K, R, t."""
        from scipy.linalg import rq
        
        # RQ decomposition of the left 3x3 part
        M = P[:, :3]
        K, R = rq(M)
        
        # Ensure positive diagonal for K
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        
        # Compute translation
        t = np.linalg.solve(K, P[:, 3])
        
        # Normalize K
        K = K / K[2, 2]
        
        return K, R, t
    
    def _get_camera_pose(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get camera intrinsics and pose for given index."""
        view_id = self.view_ids[idx]
        
        if view_id not in self._pose_cache:
            # Read projection matrix
            P = self._read_projection_matrix(self.calib_paths[idx])
            
            # Decompose into K, R, t
            K, R, t = self._decompose_projection_matrix(P)
            
            # Convert to tensors
            K_tensor = torch.from_numpy(K).float()
            R_tensor = torch.from_numpy(R).float()  
            t_tensor = torch.from_numpy(t).float()
            
            self._pose_cache[view_id] = (K_tensor, R_tensor, t_tensor)
        
        return self._pose_cache[view_id]
    
    def get_point_cloud(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get point cloud vertices and colors as tensors."""
        vertices = torch.from_numpy(self.point_cloud_vertices)
        colors = torch.from_numpy(self.point_cloud_colors)
        return vertices, colors
    
    def render_depth(self, idx: int, height: int = 1200, width: int = 1600) -> torch.Tensor:
        """Render depth map by projecting point cloud."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get camera parameters
        K, R, t = self._get_camera_pose(idx)
        K, R, t = K.to(device), R.to(device), t.to(device)
        
        # Get point cloud
        points_world, _ = self.get_point_cloud()
        points_world = points_world.to(device)
        
        # Transform to camera coordinates
        points_cam = (R @ points_world.T + t.unsqueeze(-1)).T
        
        # Keep only points in front of camera
        depth_values = points_cam[:, 2]
        front_mask = depth_values > 0
        points_cam = points_cam[front_mask]
        depth_values = depth_values[front_mask]
        
        # Project to image coordinates
        points_proj = (K @ points_cam.T).T
        u = (points_proj[:, 0] / points_proj[:, 2]).round().long()
        v = (points_proj[:, 1] / points_proj[:, 2]).round().long()
        
        # Keep points within image bounds
        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid_mask]
        v = v[valid_mask] 
        depth_values = depth_values[valid_mask]
        
        # Create depth map with z-buffering
        flat_indices = v * width + u
        depth_map_flat = torch.full((height * width,), float('inf'), device=device)
        depth_map_flat.scatter_reduce_(0, flat_indices, depth_values, reduce='amin', include_self=True)
        
        depth_map = depth_map_flat.view(height, width)
        depth_map[torch.isinf(depth_map)] = torch.nan
        
        return depth_map
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.view_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item at given index."""
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Apply transform or convert to tensor
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Get camera parameters
        K, R, t = self._get_camera_pose(idx)
        
        return {
            "image": image_tensor,
            "K": K,                    # Camera intrinsics
            "R": R,                    # Rotation matrix  
            "t": t,                    # Translation vector
            "view_id": self.view_ids[idx],
            "image_path": self.image_paths[idx],
        }


def visualize_dataset(dataset: DTUDataset, app_name: str = "DTU Visualization"):
    """Visualize DTU dataset using rerun (optional dependency)."""
    try:
        import rerun as rr
        import cv2
    except ImportError:
        raise ImportError("Please install rerun-sdk and opencv-python for visualization")
    
    rr.init(app_name, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)
    
    # Log cameras and images
    for idx in range(len(dataset)):
        sample = dataset[idx]
        
        # Load image with OpenCV
        image = cv2.imread(sample["image_path"])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Camera pose: convert from camera-to-world
        R_t = sample["R"].T  
        camera_center = -R_t @ sample["t"]
        
        rr.log(
            f"camera/{idx:03d}",
            rr.Pinhole(
                image_from_camera=sample["K"],
                resolution=(image_rgb.shape[1], image_rgb.shape[0])
            ),
            rr.Transform3D(translation=camera_center, mat3x3=R_t),
            rr.Image(image_rgb),
        )
    
    # Log point cloud
    vertices, colors = dataset.get_point_cloud()
    rr.log("point_cloud", rr.Points3D(positions=vertices.numpy(), colors=colors.numpy()))
