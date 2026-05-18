"""Synthetic dataset that yields random images for testing and development.

Each sample is a sequence of random RGB frames (clamped ``torch.randn`` in
[0, 1]) with fixed spatial resolution 504x504 and no scene geometry.
"""

import torch

from ..interface.dataset import MultiViewGeometryDataset, MultiViewTask
from ..interface.geometric_model import MultiViewInput


class FakeInput_Dataset(MultiViewGeometryDataset):
    """Map-style dataset of random image sequences.

    Args:
        length: Number of samples the dataset exposes.
        sequence_length: Number of frames per sample.
        device: Target device for returned tensors.
        seed: If not ``None``, each sample is generated from a deterministic
              per-index seed derived from this base seed, making the dataset
              reproducible.
    """

    _HEIGHT = 504
    _WIDTH  = 504

    def __init__(
        self,
        length: int,
        sequence_length: int,
        device: str | torch.device,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        self._length = length
        self._sequence_length = sequence_length
        self._device = device
        self._seed = seed

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> MultiViewTask:
        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range [0, {self._length})")

        gen = torch.Generator()
        if self._seed is not None:
            gen.manual_seed(self._seed + index)
        else:
            gen.seed()

        images = torch.randn(
            1, self._sequence_length, 3, self._HEIGHT, self._WIDTH,
            generator=gen,
        ).clamp(0.0, 1.0)

        mv_input = MultiViewInput(images=images, intrinsics=None)
        mv_input.to(device=self._device)

        return MultiViewTask(multiview_input=mv_input, scene_geometry=None)
