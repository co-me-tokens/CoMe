"""Renderer that saves per-batch point clouds as binary PLY files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..interface.geometric_model import MultiViewInput, SceneGeometry


# ==== PLY helpers ====

_PLY_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {vertex_count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "property float conf\n"
    "end_header\n"
)

_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("conf", "<f4"),
])


def _build_vertices(
    points: torch.Tensor,
    colors: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor | None,
) -> np.ndarray:
    """Flatten one batch item (S views) into a structured vertex array.

    Args:
        points: (S, 3, H, W) world-frame positions.
        colors: (S, 3, H, W) RGB in [0, 1].
        conf:   (S, 1, H, W) confidence scores.
        mask:   (S, 1, H, W) bool or None.
    """
    S, _, H, W = points.shape
    N = S * H * W

    pts_np = points.permute(0, 2, 3, 1).reshape(N, 3).numpy()
    rgb_np = (colors.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).reshape(N, 3).numpy()
    conf_np = conf.permute(0, 2, 3, 1).reshape(N).numpy()

    if mask is not None:
        valid = ~mask.permute(0, 2, 3, 1).reshape(N).numpy()
        pts_np = pts_np[valid]
        rgb_np = rgb_np[valid]
        conf_np = conf_np[valid]

    verts = np.empty(pts_np.shape[0], dtype=_VERTEX_DTYPE)
    verts["x"] = pts_np[:, 0]
    verts["y"] = pts_np[:, 1]
    verts["z"] = pts_np[:, 2]
    verts["red"] = rgb_np[:, 0]
    verts["green"] = rgb_np[:, 1]
    verts["blue"] = rgb_np[:, 2]
    verts["conf"] = conf_np
    return verts


def _write_ply(path: Path, verts: np.ndarray) -> None:
    header = _PLY_HEADER_TEMPLATE.format(vertex_count=len(verts))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())


# ==== Renderer ====

class PlyRenderer:
    """Implements RendererLike: saves one binary PLY per batch item.

    Saves files named ``{save_to}_{b:02d}.ply`` for b in 0..B-1.
    Each file merges all S views of one batch item into a single point cloud.

    Args:
        subsample_ratio: Fraction of vertices to keep per file, drawn without
            replacement.  Must be in (0, 1].  Use 1.0 to keep all vertices.
    """

    def __init__(self, *, subsample_ratio: float = 1.0) -> None:
        if not (0.0 < subsample_ratio <= 1.0):
            raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio!r}.")
        self._subsample_ratio = subsample_ratio

    def render(self, scene: SceneGeometry, input: MultiViewInput, save_to: str | Path) -> None:
        if scene.points is None:
            raise ValueError("PlyRenderer requires scene.points (got None).")
        if scene.points_conf is None:
            raise ValueError("PlyRenderer requires scene.points_conf (got None).")

        scene.to(device="cpu")
        input.to(device="cpu")

        B = input.images.shape[0]
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)

        for b in range(B):
            mask_b = scene.infer_mask[b] if scene.infer_mask is not None else None
            verts = _build_vertices(
                scene.points[b],
                input.images[b],
                scene.points_conf[b],
                mask_b,
            )
            if self._subsample_ratio < 1.0:
                k = max(1, math.ceil(len(verts) * self._subsample_ratio))
                idx = np.random.choice(len(verts), size=k, replace=False)
                verts = verts[idx]
            _write_ply(save_to.parent / f"{save_to.name}_{b:02d}.ply", verts)
