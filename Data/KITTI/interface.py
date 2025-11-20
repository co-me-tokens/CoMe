import torch
import numpy as np
import torchvision as tv
import typing as T

from PIL import Image


from pathlib import Path

from ..Interface import DepthData, DepthDataset


class KITTIDepthDataset(DepthDataset):
    def __init__(
        self, root: str | Path, sequence_length: int,
        sample_all: bool=True, use_camera: T.Literal["image_02", "image_03"]="image_02"
    ) -> None:
        super().__init__()
        root      = Path(root)
        
        raw_sequences = [d for d in root.iterdir() if d.is_dir() and (d.name.endswith("sync"))]
        self.samples  = self._sample_segments(raw_sequences, sequence_length, use_camera, sample_all)
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> DepthData:
        sample = self.samples[index]
        images = torch.stack([self._load_image(i) for i, d in sample], dim=0)
        depths = torch.stack([self._load_depth(d) for i, d in sample], dim=0)
        return DepthData(images=images.unsqueeze(0), gt_depths=depths.unsqueeze(0))
        

    @staticmethod
    def _sample_segments(sequences: list[Path], seq_length: int, use_camera: str, sample_all: bool):
        rgb_sequences = [Path(d, "image", use_camera) for d in sequences]
        dep_sequences = [Path(d, "proj_depth", "groundtruth", use_camera) for d in sequences]
        
        samples       = []
        for rgb_seq, dep_seq in zip(rgb_sequences, dep_sequences):
            rgb_files = [f for f in rgb_seq.glob("*.png") if f.is_file()]
            dep_files = [f for f in rgb_seq.glob("*.png") if f.is_file()]
        
            filename_pair = {r_f.name for r_f in rgb_files} & {d_f.name for d_f in dep_files}
            paired_files  = [(Path(rgb_seq, f), Path(dep_seq, f)) for f in sorted(list(filename_pair))]
            
            if sample_all:
                for start_idx in range(0, len(paired_files) - seq_length, seq_length):
                    samples.append(paired_files[start_idx:start_idx+seq_length])
            else:
                samples.append(paired_files[:seq_length])
        return samples
    
    @staticmethod
    def _load_depth(depth_path: Path) -> torch.Tensor:
        image_size = (320, 1024)
        
        """Load and preprocess depth map."""
        # KITTI depth maps are stored as 16-bit PNGs scaled by 256
        depth = Image.open(depth_path)
        depth_np = np.array(depth, dtype=np.float32) / 256.
        
        # Resize depth map
        depth_pil = Image.fromarray(depth_np)
        depth_resized = depth_pil.resize((image_size[1], image_size[0]), Image.NEAREST)
        depth_tensor = torch.from_numpy(np.array(depth_resized)).unsqueeze(0)
        
        return tv.transforms.CenterCrop((320, 518))(depth_tensor)

    @staticmethod
    def _load_image(image_path: Path) -> torch.Tensor:
        """Load and preprocess RGB image."""
        image_size = (320, 1024)
        
        image = Image.open(image_path).convert('RGB')
        transform = tv.transforms.Compose([
            tv.transforms.Resize(image_size),
            tv.transforms.ToTensor(),
            tv.transforms.CenterCrop((320, 518))
        ])
        
        return transform(image)
