import torch

import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class TartanAirMonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"

        self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert self.length > 0, f"No flow with '.png' suffix is found under {self.directory}"

    @staticmethod
    def load_png_format(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Failed to read image from {str(path)}")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        result = self.load_png_format(self.file_names[index])
        result = torch.tensor(result, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        result /= 255.

        return result


class TartanAirGTDepthDataset(Dataset):
    """
    Returns pixel depth in shape of (1, H, W)
    """
    def __init__(self, directory: Path, compressed: bool) -> None:
        super().__init__()
        self.directory = directory
        self.compressed = compressed
        assert self.directory.exists(), f"Depth image directory {self.directory} does not exist"

        if self.compressed:
            self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".png"]
        else:
            self.file_names = [f for f in self.directory.iterdir() if f.suffix == ".npy"]
        self.file_names.sort()
        self.length = len(self.file_names)
        assert len(self.file_names) > 0, f"No depth with '.npy' suffix is found in {self.directory}"
        
    def __len__(self): return self.length
    
    @staticmethod
    def load_npy_format(path: Path):
        return np.load(str(path))

    @staticmethod
    def load_png_format(path: Path):
        depth_rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, f"Unable to load depth image at {path}"
        depth = TartanAirGTDepthDataset.depth_rgba_float32(depth_rgba)
        return depth
    
    @staticmethod
    def depth_rgba_float32(depth_rgba):
        """
        Referenced from TartanVO codebase
        """
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def __getitem__(self, index) -> torch.Tensor:
        # Output (1, 1, H, W) tensor
        if self.compressed:
            depth_np = TartanAirGTDepthDataset.load_png_format(self.file_names[index])
        else:
            depth_np = TartanAirGTDepthDataset.load_npy_format(self.file_names[index])
        depth = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return depth
