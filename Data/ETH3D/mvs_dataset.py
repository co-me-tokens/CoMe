import logging
import numpy as np
import torch
import open3d as o3d
import pypose as pp
from pathlib import Path
from typing import Optional, Union, Callable, Dict, List, Tuple
from torchvision.io import decode_image

from ..Interface import MVSData, MVSDataset

logger = logging.getLogger(__name__)


class ETH3D_MVS_Dataset(MVSDataset):
    SCENE_NAMES = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]

    def __init__(self, root: str | Path, sequence_length: int):
        self.root = Path(root)
        self.seql = sequence_length

        self.samples = self._sample_all()

    def _sample_all(self) -> List[Dict]:
        """Sample sequences from all scenes with COLMAP data"""
        scenes_paths = [Path(self.root, scene) for scene in self.SCENE_NAMES]

        samples = []
        for scene_path in scenes_paths:
            try:
                scene_data = self._load_scene_data(scene_path)
                if scene_data is None:
                    continue

                # Sample sequences from this scene
                image_ids = list(scene_data['images'].keys())
                for start_idx in range(0, len(image_ids) - self.seql + 1, self.seql):
                    sequence_ids = image_ids[start_idx:start_idx + self.seql]
                    samples.append({
                        'scene_path': scene_path,
                        'scene_data': scene_data,
                        'sequence_ids': sequence_ids
                    })
            except Exception as e:
                logger.warning(f"Failed to load scene {scene_path.name}: {e}")
                continue

        return samples

    def _load_scene_data(self, scene_path: Path) -> Optional[Dict]:
        """Load COLMAP data for a scene"""
        calib_path = scene_path / "dslr_calibration_undistorted"
        depth_root = scene_path / "ground_truth_depth" / "dslr_images"
        image_root = scene_path / "images" / "dslr_images"

        if not all([calib_path.exists(), depth_root.exists(), image_root.exists()]):
            return None

        # Load COLMAP data
        cameras = self._load_cameras(calib_path / "cameras.txt")
        images = self._load_images(calib_path / "images.txt")
        points3d = self._load_points3d(calib_path / "points3D.txt")

        # Load reference point cloud if available
        point_cloud = self._load_reference_pointcloud(scene_path)

        # Get available depth and image files
        depth_files = {f.name: f for f in depth_root.glob("*.JPG")}
        image_files = {f.name: f for f in image_root.glob("*.JPG")}

        valid_images = {}
        for img_id, img_data in images.items():
            img_name = Path(img_data['name']).name
            if img_name in depth_files and img_name in image_files:
                valid_images[img_id] = {
                    **img_data,
                    'depth_path': depth_files[img_name],
                    'image_path': image_files[img_name]
                }

        if len(valid_images) < self.seql:
            return None

        return {
            'cameras': cameras,
            'images': valid_images,
            'points3d': points3d,
            'point_cloud': point_cloud,
            'scene_path': scene_path
        }

    def _load_cameras(self, cameras_path: Path) -> Dict:
        """Load camera intrinsics from cameras.txt"""
        cameras = {}
        with open(cameras_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(x) for x in parts[4:]]

                if model == 'PINHOLE':
                    fx, fy, cx, cy = params
                    K = torch.tensor([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=torch.float32)
                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'K': K
                    }
        return cameras

    def _load_images(self, images_path: Path) -> Dict:
        """Load image poses from images.txt"""
        images = {}
        with open(images_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue

            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]

            q = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
            q = q / torch.norm(q)  # Normalize
            R = self._quat_to_rotmat(q)
            t = torch.tensor([tx, ty, tz], dtype=torch.float32)

            images[image_id] = {
                'R': R,
                't': t,
                'camera_id': camera_id,
                'name': name
            }

            i += 2  # Skip the points2D line

        return images

    def _load_points3d(self, points3d_path: Path) -> torch.Tensor:
        """Load 3D points from points3D.txt"""
        points = []
        with open(points3d_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                points.append([x, y, z])

        return torch.tensor(points, dtype=torch.float32) if points else torch.empty(0, 3)

    def _load_reference_pointcloud(self, scene_path: Path) -> Optional[torch.Tensor]:
        """Load reference point cloud from PLY files if available"""
        scan_path = scene_path / "dslr_scan_eval"
        if not scan_path.exists():
            return None

        ply_files = list(scan_path.glob("*.ply"))
        if not ply_files:
            return None

        # Load first PLY file as reference
        try:
            pcd = o3d.io.read_point_cloud(str(ply_files[0]))
            points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
            return points
        except Exception as e:
            logger.warning(f"Failed to load point cloud from {ply_files[0]}: {e}")
            return None

    def _quat_to_rotmat(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [qw, qx, qy, qz] to rotation matrix"""
        qw, qx, qy, qz = q
        R = torch.tensor([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ], dtype=torch.float32)
        return R

    def _world2cam_pose(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert world-to-camera pose to SE3 format"""
        # Create world-to-camera transformation matrix
        T_w2c = torch.eye(4, dtype=torch.float32)
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = t

        # Convert to SE3 and get camera-to-world
        w2c_se3 = pp.from_matrix(T_w2c, ltype=pp.SE3_type)
        c2w_se3 = pp.Inv(w2c_se3)

        return c2w_se3.tensor()

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a single image"""
        try:
            image = decode_image(str(image_path)).float() / 255.0

            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3]

            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        """Load and preprocess a single depth map"""
        try:
            depth_data = np.fromfile(depth_path, dtype=np.float32)
            depth = torch.from_numpy(depth_data.reshape(4032, 6048)).float()
            depth = depth.unsqueeze(0)
            return depth
        except Exception as e:
            raise RuntimeError(f"Failed to load depth {depth_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> MVSData:
        sample = self.samples[index]
        scene_data = sample['scene_data']
        sequence_ids = sample['sequence_ids']

        # Load images and depths
        images = []
        depths = []
        poses = []
        intrinsics = []

        for img_id in sequence_ids:
            img_data = scene_data['images'][img_id]

            # Load image and depth
            image = self._load_image(img_data['image_path'])
            depth = self._load_depth(img_data['depth_path'])

            # Get pose (world2cam)
            pose = self._world2cam_pose(img_data['R'], img_data['t'])

            # Get intrinsics
            camera_id = img_data['camera_id']
            K = scene_data['cameras'][camera_id]['K']

            images.append(image)
            depths.append(depth)
            poses.append(pose)
            intrinsics.append(K)

        images_tensor = torch.stack(images, dim=0).unsqueeze(0)  # (1, S, 3, H, W)
        depths_tensor = torch.stack(depths, dim=0).unsqueeze(0)  # (1, S, 1, H, W)
        poses_tensor = torch.stack(poses, dim=0).unsqueeze(0)    # (1, S, 7)
        intrinsics_tensor = torch.stack(intrinsics, dim=0).unsqueeze(0)  # (1, S, 3, 3)

        poses_se3 = pp.SE3(poses_tensor)

        # Get point cloud (use reference point cloud if available, otherwise COLMAP points)
        if scene_data['point_cloud'] is not None:
            points = [scene_data['point_cloud']]
        else:
            points = [scene_data['points3d']]

        return MVSData(
            images=images_tensor,
            gt_depths=depths_tensor,
            points=points,
            poses=poses_se3,
            intrinsics=intrinsics_tensor
        )
