import torch
import numpy as np
from pathlib import Path
from torchvision.io import decode_image
from typing import Optional, Union, Callable
import logging

from ..Interface import DepthData, DepthDataset


logger = logging.getLogger(__name__)


class NYUdV2_Depth_Dataset(DepthDataset):
    """
    NYU Depth Dataset v2 implementation following the DepthDataset interface.
    """
    
    def __init__(
        self, 
        root: Union[str, Path], 
        transform: Optional[Callable] = None,
        depth_scale: float = 1.0,
        image_size: tuple[int, int] = (480, 640)
    ):
        """
        Initialize NYU Depth v2 dataset.
        
        Args:
            root: Root directory containing 'depth' and 'image' subdirectories
            transform: Optional transform to apply to images and depths
            depth_scale: Scale factor for depth values
            image_size: Expected image size (H, W)
        """
        self.root = Path(root)
        self.transform = transform
        self.depth_scale = depth_scale
        self.image_size = image_size
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        
        self._validate_structure()
        self._load_file_lists()
        
        logger.info(f"Initialized NYUdV2Dataset with {len(self)} samples")
    
    def _validate_structure(self) -> None:
        """Validate the expected directory structure."""
        depth_dir = self.root / "depth"
        image_dir = self.root / "image"
        
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    def _load_file_lists(self) -> None:
        """Load and sort file lists for images and depths."""
        depth_dir = self.root / "depth"
        image_dir = self.root / "image"
        
        self.depth_files = sorted(list(depth_dir.glob("*.npy")))
        self.image_files = sorted(list(image_dir.glob("*.png")))
        
        if len(self.depth_files) != len(self.image_files):
            raise ValueError(
                f"Mismatch between depth files ({len(self.depth_files)}) "
                f"and image files ({len(self.image_files)})"
            )
        
        if len(self.depth_files) == 0:
            raise ValueError(f"No data files found in {self.root}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.depth_files)
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a single image."""
        try:
            image = decode_image(str(image_path)).float() / 255.0
            
            # Ensure 3 channels (RGB)
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3]  # Drop alpha channel
            
            # Validate dimensions
            if image.shape[1:] != self.image_size:
                logger.warning(
                    f"Image size mismatch: expected {self.image_size}, "
                    f"got {image.shape[1:]} for {image_path}"
                )
            
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
    
    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        """Load and preprocess a single depth map."""
        try:
            depth = torch.from_numpy(np.load(depth_path)).float()
            
            # Apply depth scale
            depth = depth * self.depth_scale
            
            # Ensure proper shape (H, W) -> (1, H, W)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            elif depth.ndim != 3 or depth.shape[0] != 1:
                raise ValueError(f"Unexpected depth shape: {depth.shape}")
            
            # Validate dimensions
            if depth.shape[1:] != self.image_size:
                logger.warning(
                    f"Depth size mismatch: expected {self.image_size}, "
                    f"got {depth.shape[1:]} for {depth_path}"
                )
            
            return depth
        except Exception as e:
            raise RuntimeError(f"Failed to load depth {depth_path}: {e}")
    
    def __getitem__(self, index: int) -> DepthData:
        """
        Get a single image-depth pair.
        
        Args:
            index: Sample index
            
        Returns:
            DepthData with images (1, 1, 3, H, W) and gt_depths (1, 1, 1, H, W)
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")
        
        # Load single image and depth
        image = self._load_image(self.image_files[index])
        depth = self._load_depth(self.depth_files[index])
        
        # Add sequence and batch dimensions: (1, 1, 3, H, W) and (1, 1, 1, H, W)
        # Format: (B, S, C, H, W) where B=1, S=1 for single frame
        images_tensor = image.unsqueeze(0).unsqueeze(0)  # (3,H,W) -> (1,1,3,H,W)
        depths_tensor = depth.unsqueeze(0).unsqueeze(0)  # (1,H,W) -> (1,1,1,H,W)
        
        # Apply transforms if provided
        if self.transform is not None:
            images_tensor, depths_tensor = self.transform(images_tensor, depths_tensor)
        
        return DepthData(images=images_tensor, gt_depths=depths_tensor)
    
    def get_frame_paths(self, index: int) -> tuple[Path, Path]:
        """
        Get file paths for a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (image_path, depth_path) for the sample
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")
        
        return self.image_files[index], self.depth_files[index]
    
    def get_dataset_info(self) -> dict:
        """Get dataset statistics and information."""
        return {
            "root": str(self.root),
            "num_samples": len(self),
            "image_size": self.image_size,
            "depth_scale": self.depth_scale,
            "sample_image_file": str(self.image_files[0]) if self.image_files else None,
            "sample_depth_file": str(self.depth_files[0]) if self.depth_files else None,
        }


def create_nyud_v2_depth_dataset(
    root: Union[str, Path],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for NYU Depth v2 dataset.
    
    Args:
        root: Dataset root directory
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        **kwargs: Additional arguments for NYUdV2Dataset
        
    Returns:
        DataLoader configured for the dataset
    """
    dataset = NYUdV2_Depth_Dataset(root=root, **kwargs)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=DepthDataset.collate,
        pin_memory=torch.cuda.is_available()
    )
