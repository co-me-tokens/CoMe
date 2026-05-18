"""
Video (MP4/MOV) dataset loader for multi-view geometry tasks.

Given an arbitrary video, designate the max_length, step_size (default 1), and start (default 0), load
all the non-overlapping segments of video segments in the MultiviewTask format.

WARNING: This dataset holds a persistent cv2.VideoCapture handle and is NOT safe for use with
DataLoader num_workers > 0 or any multi-process / multi-threaded access pattern. It is intended
exclusively for single-process sequential inference (e.g. src/infer.py). Do not use it for
training or batched inference with worker processes.
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from ..interface.dataset import MultiViewGeometryDataset, MultiViewTask
from ..interface.geometric_model import MultiViewInput


class CustomVideo_Dataset(MultiViewGeometryDataset):
    def __init__(self, data_root: Path | str, max_length: int, device: str | torch.device, step_size: int = 1, start: int = 0):
        super().__init__()

        assert max_length >= 1, f"max_length < 1 is not reasonable. get {max_length=}"
        assert step_size  >= 1, f"step_size < 1 is not reasonable. get {step_size=}"
        assert start >= 0     , f"start < 0 is not reasonable. get {start=}"

        self.device    = device
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Could not find video file {data_root=}")

        self._cap = cv2.VideoCapture(str(self.data_root))
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture could not open {self.data_root}")

        total_frames   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_length   = max_length
        self.step_size    = step_size
        self.start        = start
        self._total_frames = total_frames
        # Number of complete non-overlapping segments fitting in the video.
        self._num_segments = max(0, (total_frames - start) // (max_length * step_size))

    def __len__(self) -> int:
        return self._num_segments

    def __getitem__(self, index: int) -> MultiViewTask:
        if index < 0 or index >= self._num_segments:
            raise IndexError(f"Index {index} out of range for dataset of length {self._num_segments}")

        first_frame_idx = self.start + index * self.max_length * self.step_size
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)

        frames: list[torch.Tensor] = []
        for j in range(self.max_length):
            target_idx = first_frame_idx + j * self.step_size
            # Seek is needed when step_size > 1 (we skip frames between reads).
            if j > 0 and self.step_size > 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ok, bgr = self._cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read frame {target_idx} from {self.data_root} "
                    f"(segment {index}, frame {j} of {self.max_length})"
                )
            # cv2 returns BGR uint8 (H, W, 3); convert to RGB float32 [0, 1] and CHW.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0)

        # Stack to [S, 3, H, W], add batch dim → [1, S, 3, H, W].
        images = torch.stack(frames).unsqueeze(0)

        mv_input = MultiViewInput(images=images, intrinsics=None)
        mv_input.to(device=self.device)

        return MultiViewTask(multiview_input=mv_input, scene_geometry=None)
