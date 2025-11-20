import torch
import pypose as pp
from .dataset import ETH3D_Depth_Dataset
from .mvs_dataset import ETH3D_MVS_Dataset
from ..Interface import DepthDataset, DepthData, MVSDataset, MVSData


class ETH3D_DepthDataset(DepthDataset):
    """ETH3D Depth Dataset interface"""
    def __init__(self, root: str, sequence_length: int):
        self.dataset = ETH3D_Depth_Dataset(root, sequence_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> DepthData:
        return self.dataset[index]


class ETH3D_MVSDataset(MVSDataset):
    """ETH3D Multi-View Stereo Dataset interface"""
    def __init__(self, root: str, sequence_length: int):
        self.dataset = ETH3D_MVS_Dataset(root, sequence_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> MVSData:
        return self.dataset[index]
