"""
Image-folder dataset loader for multi-view geometry tasks.

Given a directory of images, designate max_length, step_size (default 1), and start (default 0),
and load all non-overlapping segments in the MultiViewTask format — a direct image-folder analogue
of CustomVideo_Dataset.

Supported image extensions: .jpg, .jpeg, .png, .webp (case-insensitive). Images are ordered
lexicographically by filename (dictionary order).
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from ..interface.dataset import MultiViewGeometryDataset, MultiViewTask
from ..interface.geometric_model import MultiViewInput

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CustomImage_Dataset(MultiViewGeometryDataset):
    def __init__(
        self,
        data_root: Path | str,
        max_length: int,
        device: str | torch.device,
        step_size: int = 1,
        start: int = 0,
    ):
        super().__init__()

        assert max_length >= 1, f"max_length < 1 is not reasonable. got {max_length=}"
        assert step_size  >= 1, f"step_size < 1 is not reasonable. got {step_size=}"
        assert start      >= 0, f"start < 0 is not reasonable. got {start=}"

        self.device    = device
        self.data_root = Path(data_root)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Could not find image directory {data_root=}")

        self._image_paths = sorted(
            p for p in self.data_root.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )

        if len(self._image_paths) == 0:
            raise ValueError(
                f"No supported image files ({_SUPPORTED_EXTENSIONS}) found in {self.data_root}"
            )

        total_images      = len(self._image_paths)
        self.max_length   = max_length
        self.step_size    = step_size
        self.start        = start
        self._num_segments = max(0, (total_images - start) // (max_length * step_size))

    def __len__(self) -> int:
        return self._num_segments

    def __getitem__(self, index: int) -> MultiViewTask:
        if index < 0 or index >= self._num_segments:
            raise IndexError(f"Index {index} out of range for dataset of length {self._num_segments}")

        first_idx = self.start + index * self.max_length * self.step_size

        frames: list[torch.Tensor] = []
        for j in range(self.max_length):
            path = self._image_paths[first_idx + j * self.step_size]
            img = Image.open(path).convert("RGB")
            rgb = np.asarray(img, dtype=np.float32) / 255.0
            # (H, W, 3) → (3, H, W)
            frames.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))

        # [S, 3, H, W] → [1, S, 3, H, W]
        images = torch.stack(frames).unsqueeze(0)

        mv_input = MultiViewInput(images=images, intrinsics=None)
        mv_input.to(device=self.device)

        return MultiViewTask(multiview_input=mv_input, scene_geometry=None)
