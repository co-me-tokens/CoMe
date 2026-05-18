"""Renderer that dumps model outputs and inputs to `.npz` archives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..interface.geometric_model import MultiViewInput, SceneGeometry


# ==== Configuration ====

@dataclass(kw_only=True)
class SaveNPZRendererConfiguration:
    include_input_images: bool = True
    include_input_intrinsics: bool = True


# ==== Helpers ====

def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def _pose_convention_to_names(scene: SceneGeometry) -> list[str]:
    return [axis.name for axis in scene.pose_convention]


def _scene_archive_fields(
    scene: SceneGeometry,
    input_mvs: MultiViewInput,
    batch_idx: int,
    config: SaveNPZRendererConfiguration,
) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {
        "metadata_json": np.array(json.dumps({
            "batch_idx": batch_idx,
            "pose_convention": _pose_convention_to_names(scene),
            "has_depths": scene.depths is not None,
            "has_depths_conf": scene.depths_conf is not None,
            "has_points": scene.points is not None,
            "has_points_conf": scene.points_conf is not None,
            "has_poses": scene.poses is not None,
            "has_scene_intrinsics": scene.intrinsics is not None,
            "has_infer_mask": scene.infer_mask is not None,
            "has_input_images": config.include_input_images,
            "has_input_intrinsics": config.include_input_intrinsics and input_mvs.intrinsics is not None,
        })),
    }

    if scene.depths is not None:
        fields["scene_depths"] = _tensor_to_numpy(scene.depths[batch_idx])
    if scene.depths_conf is not None:
        fields["scene_depths_conf"] = _tensor_to_numpy(scene.depths_conf[batch_idx])
    if scene.points is not None:
        fields["scene_points"] = _tensor_to_numpy(scene.points[batch_idx])
    if scene.points_conf is not None:
        fields["scene_points_conf"] = _tensor_to_numpy(scene.points_conf[batch_idx])
    if scene.poses is not None:
        fields["scene_poses"] = _tensor_to_numpy(scene.poses[batch_idx])
    if scene.intrinsics is not None:
        fields["scene_intrinsics"] = _tensor_to_numpy(scene.intrinsics[batch_idx])
    if scene.infer_mask is not None:
        fields["scene_infer_mask"] = _tensor_to_numpy(scene.infer_mask[batch_idx])

    if config.include_input_images:
        fields["input_images"] = _tensor_to_numpy(input_mvs.images[batch_idx])
    if config.include_input_intrinsics and input_mvs.intrinsics is not None:
        fields["input_intrinsics"] = _tensor_to_numpy(input_mvs.intrinsics[batch_idx])

    return fields


# ==== Renderer ====

class SaveNPZRenderer:
    def __init__(self, config: SaveNPZRendererConfiguration | None = None) -> None:
        self.config = config if config is not None else SaveNPZRendererConfiguration()

    def render(self, scene: SceneGeometry, input: MultiViewInput, save_to: str | Path) -> None:
        scene.to(device="cpu")
        input.to(device="cpu")

        batch_size = input.images.shape[0]
        if batch_size < 1:
            raise ValueError(f"Expected at least one batch item, got batch_size={batch_size}")

        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)

        for batch_idx in range(batch_size):
            archive_fields = _scene_archive_fields(scene, input, batch_idx, self.config)
            np.savez_compressed(save_to.parent / f"{save_to.name}_{batch_idx:02d}.npz", **archive_fields)
