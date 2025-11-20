import torch
import pypose as pp
from .dataset import DTUDataset
from ..Interface import DepthDataset, DepthData, MVSDataset, MVSData, PoseDataset, PoseData


class DTU_DepthDataset(DepthDataset):
    def __init__(self, root: str, sequence_length: int):
        self.all_datasets = []
        for scan_id in range(1, 129):
            try:
                dataset = DTUDataset(root, scan_id, num_images=sequence_length)
            except:
                print(f"Failed to load dataset for {scan_id=}")
                continue
            
            self.all_datasets.append(dataset)
    
    def __len__(self) -> int:
        return len(self.all_datasets)
    
    def __getitem__(self, index) -> DepthData:
        samples = [s for s in self.all_datasets[index]]
        
        images = torch.stack([s['image'] for s in samples], dim=0)
        depths = torch.stack([
            self.all_datasets[index].render_depth(i).unsqueeze(0)
            for i in range(len(samples))
        ], dim=0)
        
        return DepthData(
            images=images.unsqueeze(0), gt_depths=depths.unsqueeze(0) / 1000.
        )


class DTU_PoseDataset(PoseDataset):
    def __init__(self, root: str, sequence_length: int):
        self.all_datasets = []
        for scan_id in range(1, 129):
            try:
                dataset = DTUDataset(root, scan_id, num_images=sequence_length)
            except:
                print(f"Failed to load dataset for {scan_id=}")
                continue
            
            self.all_datasets.append(dataset)
    
    def __len__(self) -> int:
        return len(self.all_datasets)
    
    @staticmethod
    def _opencv_to_ned_transform() -> torch.Tensor:
        """4x4 transformation matrix from OpenCV convention to NED convention.
        
        OpenCV: +x right, +y down, +z forward
        NED:    +x forward (North), +y right (East), +z down
        """
        # Rotation: x_ned = z_opencv, y_ned = x_opencv, z_ned = y_opencv
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = torch.tensor([
            [0, 0, 1],  # x_ned = z_opencv (forward)
            [1, 0, 0],  # y_ned = x_opencv (right)
            [0, 1, 0],  # z_ned = y_opencv (down)
        ], dtype=torch.float32)
        return transform
    
    def _camera_to_world_pose(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert camera pose to camera-to-world format in NED convention.
        
        Args:
            R: Rotation matrix (world-to-camera in OpenCV convention)
            t: Translation vector (world-to-camera in OpenCV convention, in mm)
            
        Returns:
            SE3 pose tensor in camera-to-world format with NED convention (in meters)
        """
        # Convert translation from mm to meters
        t_meters = t / 1000.0
        
        # Create world-to-camera SE3 pose in OpenCV convention
        T_w2c_opencv = torch.eye(4)
        T_w2c_opencv[:3, :3] = R
        T_w2c_opencv[:3, 3] = t_meters
        w2c_se3 = pp.from_matrix(T_w2c_opencv, ltype=pp.SE3_type)
        
        # Convert to camera-to-world using efficient inverse
        c2w_se3 = pp.Inv(w2c_se3)
        
        return c2w_se3.tensor()
    
    def __getitem__(self, index) -> PoseData:
        samples = [s for s in self.all_datasets[index]]
        
        images = torch.stack([s['image'] for s in samples], dim=0)
        
        # Convert poses to NED convention and camera-to-world format
        poses = []
        for s in samples:
            pose = self._camera_to_world_pose(s['R'], s['t'])
            poses.append(pose)
        
        poses_tensor = torch.stack(poses, dim=0)
        poses_se3 = pp.SE3(poses_tensor.unsqueeze(0))
        
        return PoseData(
            images=images.unsqueeze(0), poses=poses_se3
        )


class DTU_MVSDataset(MVSDataset):
    def __init__(self, root: str, sequence_length: int):
        self.all_datasets = []
        for scan_id in range(1, 129):
            try:
                dataset = DTUDataset(root, scan_id, num_images=sequence_length)
            except:
                print(f"Failed to load dataset for {scan_id=}")
                continue
            
            self.all_datasets.append(dataset)
    
    def __len__(self) -> int:
        return len(self.all_datasets)
    
    def __getitem__(self, index) -> MVSData:
        samples = [s for s in self.all_datasets[index]]
        
        images = torch.stack([s['image'] for s in samples], dim=0)
        depths = torch.stack([
            self.all_datasets[index].render_depth(i).unsqueeze(0)
            for i in range(len(samples))
        ], dim=0)
        points, _ = self.all_datasets[index].get_point_cloud() 
        T = torch.zeros(len(samples), 4, 4) 
        T[:, :3, :3] = torch.stack([s['R'] for s in samples]) 
        T[:, :3,  3] = torch.stack([s['t'] for s in samples])
        poses = pp.from_matrix(T.unsqueeze(0), ltype = pp.SE3_type) 
        intrinsics = torch.stack([s['K'] for s in samples], dim=0)
        return MVSData(
            images=images.unsqueeze(0), gt_depths=depths.unsqueeze(0), points=[points], poses=poses, intrinsics=intrinsics.unsqueeze(0)
        )
