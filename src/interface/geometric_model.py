import math
import typing as T

import torch
import pypose as pp
import torch.nn.functional as F
import jaxtyping as jt
from enum import Enum, auto
from beartype import beartype

from dataclasses import dataclass
from typing import Protocol, TypeAlias
from typing_extensions import Self


def _collate_optional_tensor(values: list[torch.Tensor | None], field_name: str) -> torch.Tensor | None:
    if all(v is None for v in values):
        return None
    if all(v is not None for v in values):
        return torch.cat(T.cast(list[torch.Tensor], values), dim=0)
    raise ValueError(
        f"Non-uniform batch for {field_name}: some samples provide tensors while others provide None."
    )


def _smart_resize_spatial(tensor: torch.Tensor, int_h: int, int_w: int, tgt_h: int, tgt_w: int, mode: str) -> torch.Tensor:
    """Scale a [B, S, C, H, W] tensor to (int_h, int_w) then center-crop to (tgt_h, tgt_w)."""
    B, S, C, H, W = tensor.shape
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    scaled = F.interpolate(
        tensor.reshape(B * S, C, H, W), size=(int_h, int_w), mode=mode, **kwargs
    ).reshape(B, S, C, int_h, int_w)
    top  = (int_h - tgt_h) // 2
    left = (int_w - tgt_w) // 2
    return scaled[..., top:top + tgt_h, left:left + tgt_w]


class _TensorMovable:
    """Mixin that moves all tensor fields to a given device/dtype."""

    def to(self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        for key in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                object.__setattr__(self, key, value.to(device=device, dtype=dtype))
        return self

    def reshape_batch(self, *dims: int):
        for key in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                x = value.reshape(*dims, *value.shape[len(dims):])
                object.__setattr__(self, key, x)
        return self


@jt.jaxtyped(typechecker=beartype)
@dataclass(slots=True)
class TensorProbe(_TensorMovable):
    data: jt.Float[torch.Tensor, "..."] | None = None


@jt.jaxtyped(typechecker=beartype)
@dataclass(kw_only=True, slots=True, eq=False)
class MultiViewInput(_TensorMovable):
    images    : jt.Float32[torch.Tensor, "B S 3 H W"]
    intrinsics: jt.Float32[torch.Tensor, "B S 3 3"] | None
    
    @staticmethod
    def collate(batch: list["MultiViewInput"]) -> "MultiViewInput":
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty MultiViewInput batch.")
        return MultiViewInput(
            images=torch.cat([b.images for b in batch], dim=0),
            intrinsics=_collate_optional_tensor(
                [b.intrinsics for b in batch], "MultiViewInput.intrinsics"
            ),
        )

    def resize(self, height: int, width: int) -> Self:
        """Smart resize: uniform scale to cover target, then center crop. Preserves aspect ratio."""
        _, _, _, H, W = self.images.shape
        scale = min(H / height, W / width)
        int_h, int_w = math.ceil(H / scale), math.ceil(W / scale)

        object.__setattr__(self, "images", _smart_resize_spatial(self.images, int_h, int_w, height, width, "bilinear"))

        if self.intrinsics is not None:
            K = self.intrinsics.clone()
            K[..., 0, :] /= W / int_w
            K[..., 1, :] /= H / int_h
            K[..., 0, 2] -= (int_w - width)  / 2.0
            K[..., 1, 2] -= (int_h - height) / 2.0
            object.__setattr__(self, "intrinsics", K)

        return self


class PoseConvention(Enum):
    N = F = auto()
    S = B = auto()
    E = R = auto()
    W = L = auto()
    U = auto()
    D = auto()


CoordinateConvention: TypeAlias = tuple[PoseConvention, PoseConvention, PoseConvention]

_PAIR_NAMES = ("Forward/Backward", "Right/Left", "Up/Down")


def _axis_pair(d: PoseConvention) -> int:
    """Axis pair index: 0=Forward/Backward, 1=Right/Left, 2=Up/Down."""
    return (d.value - 1) // 2


def _axis_sign(d: PoseConvention) -> int:
    """Positive direction (+1 for F,R,U) or negative (-1 for B,L,D)."""
    return 1 if d.value % 2 == 1 else -1


def _validate_convention(conv: CoordinateConvention) -> None:
    pairs = {_axis_pair(d) for d in conv}
    if len(pairs) != 3:
        labels = ", ".join(_PAIR_NAMES[p] for p in sorted(pairs))
        raise ValueError(
            f"Invalid coordinate convention {conv}: axes must span all three "
            f"axis pairs (Forward/Backward, Right/Left, Up/Down), got only ({labels})."
        )


def _conversion_matrix(
    src: CoordinateConvention, tgt: CoordinateConvention, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build the 3x3 signed permutation matrix that converts coordinates from *src* to *tgt*."""
    C = torch.zeros(3, 3, device=device, dtype=dtype)
    for j, b in enumerate(tgt):
        for i, a in enumerate(src):
            if _axis_pair(a) == _axis_pair(b):
                C[j, i] = float(_axis_sign(b) * _axis_sign(a))
    return C


@jt.jaxtyped(typechecker=beartype)
@dataclass(kw_only=True, slots=True, eq=False)
class SceneGeometry(_TensorMovable):
    """Multi-view scene geometry with documented coordinate frames.

    Coordinate frames:
        pose_convention defines the axis labeling (X, Y, Z directions) shared by
        all spatially-oriented fields.  ``to(convention=...)`` re-expresses every
        field consistently.

        poses      -- Camera-to-world SE3 transforms [tx, ty, tz, qx, qy, qz, qw].
                      Translation = camera origin in world frame.
                      Quaternion (pypose xyzw) rotates camera-local vectors into world frame.
        points     -- Per-pixel 3D positions in **world frame**.
        depths     -- Positive distance along the camera optical axis.
                      Scalar, convention-independent.
        intrinsics -- Camera-local projection matrices (3D camera-local -> pixels).
                      Converted jointly with pose_convention.
        depths_conf, points_conf, infer_mask -- No coordinate frame.
    """

    pose_convention: CoordinateConvention

    depths     : jt.Float32[torch.Tensor, "B S 1 H W"] | None  # convention-independent scalar
    depths_conf: jt.Float32[torch.Tensor, "B S 1 H W"] | None

    points     : jt.Float32[torch.Tensor, "B S 3 H W"] | None  # world frame
    points_conf: jt.Float32[torch.Tensor, "B S 1 H W"] | None

    poses      : jt.Float32[torch.Tensor, "B S 7"]     | None  # camera-to-world [t | q_xyzw]
    intrinsics : jt.Float32[torch.Tensor, "B S 3 3"]   | None  # camera-local

    infer_mask : jt.Bool[torch.Tensor, "B S 1 H W"]    | None
    
    @staticmethod
    def collate(batch: list["SceneGeometry"]) -> "SceneGeometry":
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty SceneGeometry batch.")
        pose_convention = batch[0].pose_convention
        if not all(g.pose_convention == pose_convention for g in batch):
            raise ValueError(
                "Non-uniform batch for SceneGeometry.pose_convention: all samples must share one convention."
            )
        _c = _collate_optional_tensor
        return SceneGeometry(
            pose_convention=pose_convention,
            depths      = _c([g.depths      for g in batch], "SceneGeometry.depths"),
            depths_conf = _c([g.depths_conf for g in batch], "SceneGeometry.depths_conf"),
            points      = _c([g.points      for g in batch], "SceneGeometry.points"),
            points_conf = _c([g.points_conf for g in batch], "SceneGeometry.points_conf"),
            poses       = _c([g.poses       for g in batch], "SceneGeometry.poses"),
            intrinsics  = _c([g.intrinsics  for g in batch], "SceneGeometry.intrinsics"),
            infer_mask  = _c([g.infer_mask  for g in batch], "SceneGeometry.infer_mask"),
        )

    def resize(self, height: int, width: int) -> Self:
        """Smart resize: uniform scale to cover target, then center crop. Preserves aspect ratio."""
        spatial = self.depths if self.depths is not None else (self.points if self.points is not None else self.infer_mask)
        if spatial is None:
            return self

        _, _, _, H, W = spatial.shape
        scale = min(H / height, W / width)
        int_h, int_w = math.ceil(H / scale), math.ceil(W / scale)

        if self.depths is not None:
            object.__setattr__(self, "depths", _smart_resize_spatial(self.depths, int_h, int_w, height, width, "bicubic"))
        if self.depths_conf is not None:
            object.__setattr__(self, "depths_conf", _smart_resize_spatial(self.depths_conf, int_h, int_w, height, width, "nearest"))
        if self.points is not None:
            object.__setattr__(self, "points", _smart_resize_spatial(self.points, int_h, int_w, height, width, "nearest"))
        if self.points_conf is not None:
            object.__setattr__(self, "points_conf", _smart_resize_spatial(self.points_conf, int_h, int_w, height, width, "nearest"))
        if self.infer_mask is not None:
            object.__setattr__(self, "infer_mask", _smart_resize_spatial(self.infer_mask.float(), int_h, int_w, height, width, "nearest").bool())

        if self.intrinsics is not None:
            K = self.intrinsics.clone()
            K[..., 0, :] /= W / int_w
            K[..., 1, :] /= H / int_h
            K[..., 0, 2] -= (int_w - width)  / 2.0
            K[..., 1, 2] -= (int_h - height) / 2.0
            object.__setattr__(self, "intrinsics", K)

        return self

    def to(self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None, convention: CoordinateConvention | None = None):
        _TensorMovable.to(self, device=device, dtype=dtype)
        if convention is None:
            return self

        _validate_convention(convention)
        if convention == self.pose_convention:
            return self

        ref = self.poses if self.poses is not None else (self.points if self.points is not None else self.intrinsics)
        if ref is None:
            object.__setattr__(self, "pose_convention", convention)
            return self

        C = _conversion_matrix(
            self.pose_convention, convention,
            device=ref.device, dtype=ref.dtype,
        )

        if self.points is not None:
            B, S, _, H, W = self.points.shape
            pts = self.points.reshape(B, S, 3, H * W)
            object.__setattr__(self, "points", (C @ pts).reshape(B, S, 3, H, W))

        if self.poses is not None:
            t = self.poses[..., :3]
            R = pp.SO3(self.poses[..., 3:]).matrix()
            t_new = (C @ t.unsqueeze(-1)).squeeze(-1)
            q_new = pp.from_matrix(C @ R @ C.T, ltype=pp.SO3_type)
            object.__setattr__(self, "poses", torch.cat([t_new, q_new], dim=-1))

        if self.intrinsics is not None:
            object.__setattr__(self, "intrinsics", self.intrinsics @ C.T)

        object.__setattr__(self, "pose_convention", convention)
        return self


class GeometricPredictorLike(Protocol):
    """Structural type for geometric predictors.

    Implementations are expected to be concrete ``torch.nn.Module`` subclasses (a protocol cannot
    inherit from ``nn.Module`` under static checking rules).
    """

    def __call__(self, input: MultiViewInput) -> SceneGeometry: ...

    def inject_probe(self, block_index: int, probe: TensorProbe) -> None:
        """
        Properly register the forward hook such that the probe.data is filled with the up-to-date
        feature *after* inferencing block_index in the model for each inference.
        """
        ...

    @property
    def patch_start_index(self) -> int:
        """
        Return the number of prepended tokens before the image patches.
        """
        ...

    # Subset of ``torch.nn.Module`` used on predictors (``.eval()``, ``.cuda()``, ``.to(...)``).
    def eval(self) -> Self: ...

    def cuda(self, device: int | torch.device | None = None) -> Self: ...

    def to(self, *args: T.Any, **kwargs: T.Any) -> Self: ...
