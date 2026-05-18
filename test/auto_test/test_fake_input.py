"""Tests for the FakeInput_Dataset synthetic loader."""

import pytest
import torch

from src.dataset.fake_input import FakeInput_Dataset
from src.interface.dataset import MultiViewTask


DEVICE = "cpu"
LENGTH = 5
SEQ_LEN = 4


# ==== Construction validation ====

class TestFakeInputValidation:
    def test_invalid_length(self):
        with pytest.raises(ValueError, match="length must be >= 1"):
            FakeInput_Dataset(length=0, sequence_length=2, device=DEVICE)

    def test_invalid_sequence_length(self):
        with pytest.raises(ValueError, match="sequence_length must be >= 1"):
            FakeInput_Dataset(length=5, sequence_length=0, device=DEVICE)


# ==== Dataset behaviour ====

class TestFakeInputDataset:
    def test_length(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        assert len(ds) == LENGTH

    def test_output_shape(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        task = ds[0]
        assert task.multiview_input.images.shape == (1, SEQ_LEN, 3, 504, 504)

    def test_pixel_range(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE, seed=42)
        task = ds[0]
        assert task.multiview_input.images.min() >= 0.0
        assert task.multiview_input.images.max() <= 1.0

    def test_intrinsics_none(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        task = ds[0]
        assert task.multiview_input.intrinsics is None

    def test_scene_geometry_none(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        task = ds[0]
        assert task.scene_geometry is None

    def test_device_cpu(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        task = ds[0]
        assert task.multiview_input.images.device.type == "cpu"

    def test_index_out_of_range(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        with pytest.raises(IndexError):
            ds[LENGTH]
        with pytest.raises(IndexError):
            ds[-1]

    def test_various_sequence_lengths(self):
        for seq in (1, 2, 8, 32):
            ds = FakeInput_Dataset(length=1, sequence_length=seq, device=DEVICE)
            task = ds[0]
            assert task.multiview_input.images.shape[1] == seq

    def test_seed_reproducibility(self):
        ds_a = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE, seed=7)
        ds_b = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE, seed=7)
        assert torch.equal(ds_a[2].multiview_input.images, ds_b[2].multiview_input.images)

    def test_collate_batch(self):
        ds = FakeInput_Dataset(length=LENGTH, sequence_length=SEQ_LEN, device=DEVICE)
        batch = [ds[0], ds[1]]
        collated = MultiViewTask.collate(batch)
        assert collated.multiview_input.images.shape == (2, SEQ_LEN, 3, 504, 504)
        assert collated.scene_geometry is None
