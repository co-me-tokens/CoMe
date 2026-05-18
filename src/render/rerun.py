"""Render SceneGeometry + MultiViewInput into a coherent 3D rerun visualization.

Entity hierarchy (all converted to RDF before logging):

    /                            ViewCoordinates.RDF  +  AnnotationContext
    camera/{s}/                  Transform3D              (camera-to-world)
    camera/{s}/pinhole/          Pinhole
    camera/{s}/pinhole/rgb/      Image
    camera/{s}/pinhole/depth/    DepthImage
    camera/{s}/pinhole/mask/     SegmentationImage        (infer_mask dimming)
    point_cloud/{s}/             Points3D                 (world frame)

``RerunVideoRenderer`` uses fixed paths ``camera/...`` and ``point_cloud`` and a
``frame`` timeline (``sequence`` = global frame index) so scrubbing time plays
back sequential outputs instead of stacking one entity per slot ``s``.
"""

import torch
import rerun as rr
import numpy as np
import rerun.blueprint as rrb
import jaxtyping as jt
from enum import Enum, auto
from beartype import beartype
from dataclasses import dataclass
from pathlib import Path

from ..utility.diagnostic import Diagnostics
from ..interface.geometric_model import SceneGeometry, MultiViewInput, PoseConvention

_RDF = (PoseConvention.R, PoseConvention.D, PoseConvention.F)
_MERGE_CLASS_ID = 1
_ORIG_CLASS_ID  = 0


# ==== Configuration ====

@jt.jaxtyped(typechecker=beartype)
@dataclass
class RerunRendererConfiguration:
    application_id      : str   = "scene_geometry"
    point_radius        : float = -1.0
    image_plane_distance: float = 0.05
    mask_opacity        : float = 0.50
    point_subsample_rate: float = 0.0
    min_confidence      : float = 1.001


# ==== Renderer ====

class RerunRenderer:

    class Mode(Enum):
        File   = auto()
        Stream = auto()

    def __init__(self, config: RerunRendererConfiguration | None = None):
        self.config = config if config is not None else RerunRendererConfiguration()

    # ---- public API ---------------------------------------------------------

    def render(
        self,
        scene:   SceneGeometry,
        input:   MultiViewInput,
        save_to: Path | str,
        mode:    Mode = Mode.File,
    ):
        rr.init(self.config.application_id)

        match mode:
            case self.Mode.File:   rr.save(str(Path(save_to).with_suffix(".rrd")))
            case self.Mode.Stream: rr.connect_grpc()

        self._log(scene, input)
        rr.disconnect()
        
        Diagnostics.log(f"Rerun Renderer save result to {save_to}, save_mode={mode}")

    # ---- core logging -------------------------------------------------------

    def _log(self, scene: SceneGeometry, input: MultiViewInput):
        scene.to(device="cpu", convention=_RDF)

        images     = input.images.cpu()
        intrinsics = scene.intrinsics
        if intrinsics is None:
            if input.intrinsics is None:
                raise ValueError("Neither SceneGeometry nor MultiViewInput provides intrinsics.")
            intrinsics = input.intrinsics.cpu()

        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        rr.log("/", rr.AnnotationContext([
            (_MERGE_CLASS_ID, "Merged", (255, 0, 0)),
            (_ORIG_CLASS_ID , "Untouched", (0, 0, 0))
        ]), static=True)

        B, S = images.shape[:2]
        rr.send_blueprint(self._blueprint(S, has_depth=scene.depths is not None))

        for b in range(B):
            rr.set_time("batch", sequence=b)
            for s in range(S):
                self._log_camera(b, s, images, intrinsics, scene)
                self._log_points(b, s, images, scene)

    # ---- per-view entities --------------------------------------------------

    def _log_camera(
        self, b: int, s: int,
        images: torch.Tensor, intrinsics: torch.Tensor, scene: SceneGeometry,
    ):
        cam     = f"camera/{s}"
        pinhole = f"{cam}/pinhole"
        

        if scene.poses is not None:
            rr.log(cam, rr.Transform3D(
                translation=scene.poses[b, s, :3].numpy(),
                quaternion=rr.Quaternion(xyzw=scene.poses[b, s, 3:].numpy()),
            ))

        W, H = images.shape[-1], images.shape[-2]
        rr.log(pinhole, rr.Pinhole(
            image_from_camera=intrinsics[b, s].numpy(),
            resolution=[W, H],
            image_plane_distance=self.config.image_plane_distance,
        ))

        rr.log(f"{pinhole}/rgb", rr.Image(images[b, s].permute(1, 2, 0).contiguous().numpy()))

        if scene.depths is not None:
            rr.log(f"{pinhole}/depth", rr.DepthImage(scene.depths[b, s, 0].numpy(), meter=1.0))

        if scene.infer_mask is not None:
            seg = scene.infer_mask[b, s, 0].numpy().astype(np.uint8) * _MERGE_CLASS_ID
            rr.log(f"{pinhole}/mask", rr.SegmentationImage(seg, opacity=self.config.mask_opacity))

    def _log_points(
        self, b: int, s: int,
        images: torch.Tensor, scene: SceneGeometry,
    ):
        if scene.points is None:
            return

        positions = scene.points[b, s].permute(1, 2, 0).reshape(-1, 3)

        keep = torch.ones(positions.shape[0], dtype=torch.bool)
        if scene.points_conf is not None:
            keep &= scene.points_conf[b, s, 0].reshape(-1) > self.config.min_confidence
        if scene.infer_mask is not None:
            keep &= ~scene.infer_mask[b, s, 0].reshape(-1)

        if self.config.point_subsample_rate > 0:
            keep &= torch.rand(keep.shape[0]) > self.config.point_subsample_rate

        positions = positions[keep]
        if positions.numel() == 0:
            return

        colors = images[b, s].permute(1, 2, 0).reshape(-1, 3)[keep]
        rr.log(f"point_cloud/{s}", rr.Points3D(
            positions=positions.numpy(),
            colors=colors.numpy(),
            radii=self.config.point_radius,
        ))

    # ---- blueprint ----------------------------------------------------------

    @staticmethod
    def _blueprint(num_views: int, *, has_depth: bool) -> rrb.Blueprint:
        """Vertical 3:1 -- 3D scene on top, per-view 2D tabs on bottom."""
        tabs: list[rrb.Container] = []

        for s in range(num_views):
            pinhole = f"camera/{s}/pinhole"
            rgb_view = rrb.Spatial2DView(
                origin=pinhole, name=f"RGB {s}",
                contents=["$origin/**", "-$origin/depth/**"],
            )

            if has_depth:
                depth_view = rrb.Spatial2DView(
                    origin=pinhole, name=f"Depth {s}",
                    contents=["$origin/**", "-$origin/rgb/**"],
                )
                tabs.append(rrb.Horizontal(rgb_view, depth_view, column_shares=[1, 1], name=f"View {s}"))
            else:
                tabs.append(rrb.Horizontal(rgb_view, column_shares=[1], name=f"View {s}"))

        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="/", name="3D Scene",
                    contents=["$origin/**"] + [
                        f"-$origin/camera/{idx}/pinhole/depth"
                        for idx in range(num_views)
                    ]
                ),
                rrb.Tabs(*tabs),
                row_shares=[3, 1],
            ),
            collapse_panels=True,
        )


# ==== Video timeline renderer (single entity, frame axis) ====

_FRAME_TIMELINE = "frame"
_VIDEO_CAM = "camera"
_VIDEO_PINHOLE = f"{_VIDEO_CAM}/pinhole"
_VIDEO_POINT_CLOUD = "point_cloud"
_VIDEO_TRAJECTORY = "camera_trajectory"


def _radiant_colors(n: int) -> np.ndarray:
    """(n, 3) uint8 RGB sweeping the visible spectrum (red -> violet)."""
    h = np.linspace(0.0, 0.85, n)
    r = np.clip(np.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = np.clip(2.0 - np.abs(h * 6.0 - 2.0), 0.0, 1.0)
    b = np.clip(2.0 - np.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class RerunVideoRenderer:
    """Log ``[B, S, ...]`` geometry as one camera + one point cloud over time.

    For each pair ``(b, s)`` logs to the same entity paths while setting
    ``rr.set_time(_FRAME_TIMELINE, sequence=b * S + s)``, so the Rerun viewer
    timeline scrubs through frames. Use when ``S`` is a temporal stack (e.g.
    after ``reshape_batch(1, num_frames)``) rather than simultaneous multi-view.
    """

    def __init__(self, config: RerunRendererConfiguration | None = None):
        self.config = config if config is not None else RerunRendererConfiguration()

    def render(
        self,
        scene: SceneGeometry,
        input: MultiViewInput,
        save_to: Path | str,
        mode: RerunRenderer.Mode = RerunRenderer.Mode.File,
    ) -> None:
        rr.init(self.config.application_id)

        match mode:
            case RerunRenderer.Mode.File:
                rr.save(str(Path(save_to).with_suffix(".rrd")))
            case RerunRenderer.Mode.Stream:
                rr.connect_grpc()

        self._log(scene, input)
        rr.disconnect()

        Diagnostics.log(f"RerunVideoRenderer save result to {save_to}, save_mode={mode}")

    def _log(self, scene: SceneGeometry, input: MultiViewInput) -> None:
        scene.to(device="cpu", convention=_RDF)

        images = input.images.cpu()
        intrinsics = scene.intrinsics
        if intrinsics is None:
            if input.intrinsics is None:
                raise ValueError("Neither SceneGeometry nor MultiViewInput provides intrinsics.")
            intrinsics = input.intrinsics.cpu()

        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        rr.log(
            "/",
            rr.AnnotationContext(
                [
                    (_MERGE_CLASS_ID, "Merged", (255, 0, 0)),
                    (_ORIG_CLASS_ID, "Untouched", (0, 0, 0)),
                ]
            ),
            static=True,
        )

        B, S = images.shape[:2]
        rr.send_blueprint(self._blueprint_video(has_depth=scene.depths is not None))

        traj_positions, traj_colors = self._prepare_trajectory(scene, B, S)

        for b in range(B):
            for s in range(S):
                frame_idx = b * S + s
                rr.set_time(_FRAME_TIMELINE, sequence=frame_idx)
                self._log_camera_frame(b, s, images, intrinsics, scene)
                self._log_points_frame(b, s, images, scene)
                self._log_trajectory_at(traj_positions, traj_colors, frame_idx)

    def _prepare_trajectory(
        self, scene: SceneGeometry, B: int, S: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if scene.poses is None:
            return None, None

        positions = np.stack([
            scene.poses[b, s, :3].numpy()
            for b in range(B) for s in range(S)
        ])
        if positions.shape[0] < 2:
            return None, None

        colors = _radiant_colors(positions.shape[0] - 1)
        return positions, colors

    def _log_trajectory_at(
        self,
        positions: np.ndarray | None,
        colors: np.ndarray | None,
        frame_idx: int,
    ) -> None:
        if positions is None or colors is None or frame_idx < 1:
            return

        segments = [[positions[i], positions[i + 1]] for i in range(frame_idx)]
        rr.log(
            _VIDEO_TRAJECTORY,
            rr.LineStrips3D(strips=segments, colors=colors[:frame_idx]),
        )

    def _log_camera_frame(
        self,
        b: int,
        s: int,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        scene: SceneGeometry,
    ) -> None:
        if scene.poses is not None:
            rr.log(
                _VIDEO_CAM,
                rr.Transform3D(
                    translation=scene.poses[b, s, :3].numpy(),
                    quaternion=rr.Quaternion(xyzw=scene.poses[b, s, 3:].numpy()),
                ),
            )

        W, H = images.shape[-1], images.shape[-2]
        rr.log(
            _VIDEO_PINHOLE,
            rr.Pinhole(
                image_from_camera=intrinsics[b, s].numpy(),
                resolution=[W, H],
                image_plane_distance=self.config.image_plane_distance,
            ),
        )

        rr.log(f"{_VIDEO_PINHOLE}/rgb", rr.Image(images[b, s].permute(1, 2, 0).contiguous().numpy()))

        if scene.depths is not None:
            rr.log(f"{_VIDEO_PINHOLE}/depth", rr.DepthImage(scene.depths[b, s, 0].numpy(), meter=1.0))

        if scene.infer_mask is not None:
            seg = scene.infer_mask[b, s, 0].numpy().astype(np.uint8) * _MERGE_CLASS_ID
            rr.log(f"{_VIDEO_PINHOLE}/mask", rr.SegmentationImage(seg, opacity=self.config.mask_opacity))

    def _log_points_frame(
        self,
        b: int,
        s: int,
        images: torch.Tensor,
        scene: SceneGeometry,
    ) -> None:
        if scene.points is None:
            return

        positions = scene.points[b, s].permute(1, 2, 0).reshape(-1, 3)

        keep = torch.ones(positions.shape[0], dtype=torch.bool)
        if scene.points_conf is not None:
            keep &= scene.points_conf[b, s, 0].reshape(-1) > self.config.min_confidence
        if scene.infer_mask is not None:
            keep &= ~scene.infer_mask[b, s, 0].reshape(-1)

        if self.config.point_subsample_rate > 0:
            keep &= torch.rand(keep.shape[0]) > self.config.point_subsample_rate

        positions = positions[keep]
        if positions.numel() == 0:
            return

        colors = images[b, s].permute(1, 2, 0).reshape(-1, 3)[keep]
        rr.log(
            _VIDEO_POINT_CLOUD,
            rr.Points3D(
                positions=positions.numpy(),
                colors=colors.numpy(),
                radii=self.config.point_radius,
            ),
        )

    @staticmethod
    def _blueprint_video(*, has_depth: bool) -> rrb.Blueprint:
        """Single pinhole origin; 3D view + one video strip (RGB [+ depth])."""
        rgb_view = rrb.Spatial2DView(
            origin=_VIDEO_PINHOLE,
            name="RGB",
            contents=["$origin/**", "-$origin/depth/**"],
        )
        if has_depth:
            depth_view = rrb.Spatial2DView(
                origin=_VIDEO_PINHOLE,
                name="Depth",
                contents=["$origin/**", "-$origin/rgb/**"],
            )
            bottom = rrb.Horizontal(rgb_view, depth_view, column_shares=[1, 1], name="Video")
        else:
            bottom = rrb.Horizontal(rgb_view, column_shares=[1], name="Video")

        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="/",
                    name="3D Scene",
                    contents=["$origin/**", f"-$origin/{_VIDEO_PINHOLE}/depth"],
                ),
                bottom,
                row_shares=[3, 1],
            ),
            collapse_panels=True,
        )
