"""DTU Dataset - Clean implementation for multi-view stereo depth estimation."""

from .dataset   import DTUDataset
from .interface import DTU_DepthDataset, DTU_PoseDataset, DTU_MVSDataset

default_root = '{YOUR_PATH_TO}/DTU_MVS'
