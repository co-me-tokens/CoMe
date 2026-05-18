"""TartanAir v1 dataset loader for multi-view geometry tasks."""

import numpy as np
import torch
import pypose as pp
from enum import Enum, auto
from pathlib import Path
from PIL import Image
from rich.progress import track

from ..interface.dataset import MultiViewGeometryDataset, MultiViewTask
from ..interface.geometric_model import MultiViewInput, SceneGeometry, PoseConvention

_H, _W = 480, 640

_TARTAN_AIR_INTRINSICS = torch.tensor([
    [320.0,   0.0, 320.0],
    [  0.0, 320.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=torch.float32)


class TartanAirv1_Dataset(MultiViewGeometryDataset):

    class Camera(Enum):
        Left  = auto()
        Right = auto()

    def __init__(self, data_root: Path | str, segment_length: int, step: int, device: torch.device | str, camera: Camera = Camera.Left) -> None:
        super().__init__()

        if segment_length < 1:
            raise ValueError(f"segment_length must be >= 1, got {segment_length}")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        data_root = Path(data_root)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {data_root}")

        self._data_root = data_root
        self._segment_length = segment_length
        self._step = step
        self._device = device

        side = "left" if camera == self.Camera.Left else "right"
        self._side = side

        traj_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("P"))
        if not traj_dirs:
            raise FileNotFoundError(f"No trajectory directories (P*) found in {data_root}")

        self._traj_dirs: list[Path] = []
        self._traj_poses: list[np.ndarray] = []
        self._segments: list[tuple[int, int]] = []

        effective_length = (segment_length - 1) * step + 1
        stride = segment_length * step

        for traj_dir in traj_dirs:
            img_dir = traj_dir / f"image_{side}"
            pose_file = traj_dir / f"pose_{side}.txt"

            if not img_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            if not pose_file.is_file():
                raise FileNotFoundError(f"Pose file not found: {pose_file}")

            n_frames = len(list(img_dir.glob("*.png")))
            poses = np.loadtxt(pose_file, ndmin=2)

            if poses.shape[0] != n_frames:
                raise ValueError(f"Pose count ({poses.shape[0]}) != image count ({n_frames}) in {traj_dir}")

            if n_frames < effective_length:
                continue

            traj_idx = len(self._traj_dirs)
            self._traj_dirs.append(traj_dir)
            self._traj_poses.append(poses)

            n_segments = (n_frames - effective_length) // stride + 1
            for j in range(n_segments):
                self._segments.append((traj_idx, j * stride))

        v, u = torch.meshgrid(
            torch.arange(_H, dtype=torch.float32),
            torch.arange(_W, dtype=torch.float32),
            indexing="ij",
        )
        self._pixels = torch.stack([u, v], dim=-1).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> MultiViewTask:
        traj_idx, start = self._segments[index]
        traj_dir = self._traj_dirs[traj_idx]
        poses_all = self._traj_poses[traj_idx]

        S = self._segment_length
        frame_indices = [start + i * self._step for i in range(S)]

        images: list[torch.Tensor] = []
        for fi in frame_indices:
            path = traj_dir / f"image_{self._side}" / f"{fi:06d}_{self._side}.png"
            arr = np.array(Image.open(path))
            images.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
        images_t = torch.stack(images).unsqueeze(0)

        depths: list[torch.Tensor] = []
        for fi in frame_indices:
            path = traj_dir / f"depth_{self._side}" / f"{fi:06d}_{self._side}_depth.png"
            rgba = np.array(Image.open(path))
            depth = rgba.view(np.float32)[:, :, 0].copy()
            depths.append(torch.from_numpy(depth))
        depths_t = torch.stack(depths).unsqueeze(0).unsqueeze(2)

        raw_poses = torch.from_numpy(poses_all[frame_indices].copy()).float()
        world_poses = pp.SE3(raw_poses)
        relative_poses = world_poses[0].Inv() @ world_poses  # type: ignore[attr-defined]
        relative_poses_t = relative_poses.tensor().unsqueeze(0)

        K = _TARTAN_AIR_INTRINSICS.unsqueeze(0).expand(S, -1, -1).clone().unsqueeze(0)

        points_list: list[torch.Tensor] = []
        for s_idx in range(S):
            depth_flat = depths[s_idx].reshape(-1)
            pts_cam = pp.pixel2point(self._pixels, depth_flat, _TARTAN_AIR_INTRINSICS)
            pts_frame0 = relative_poses[s_idx].Act(pts_cam)
            points_list.append(pts_frame0.reshape(_H, _W, 3).permute(2, 0, 1))
        points_t = torch.stack(points_list).unsqueeze(0)

        mv_input = MultiViewInput(images=images_t, intrinsics=K.clone())
        scene_geo = SceneGeometry(
            pose_convention=(PoseConvention.R, PoseConvention.D, PoseConvention.F),
            depths=depths_t,
            depths_conf=None,
            points=points_t,
            points_conf=None,
            poses=relative_poses_t,
            intrinsics=K,
            infer_mask=None
        )

        mv_input.to(device=self._device)
        scene_geo.to(device=self._device)

        return MultiViewTask(multiview_input=mv_input, scene_geometry=scene_geo)


_TARTAN_AIR_ENVS = [
    "abandonedfactory",
    "amusement",
    "endofworld",
    "hongkongalley",
    "house",
    "japanesealley",
    "office",
    "oldtown",
    "seasonsforest",
    "slaughter",
    "westerndesert",
    "abandonedfactory_night",
    "carwelding",
    "gascola",
    "hospital",
    "neighborhood",
    "ocean",
    "office2",
    "seasidetown",
    "seasonsforest_winter",
    "soulcity",
]


def AllTartanAirv1_Dataset(
    data_root: Path | str,
    segment_length: int,
    step: int,
    device: torch.device | str,
    camera: TartanAirv1_Dataset.Camera = TartanAirv1_Dataset.Camera.Left,
) -> torch.utils.data.ConcatDataset[MultiViewTask]:
    return torch.utils.data.ConcatDataset([
        TartanAirv1_Dataset(Path(data_root, env, "Data"), segment_length, step, device, camera)
        for env in track(_TARTAN_AIR_ENVS, description="Loading Full TartanAir-v1 Dataset...")
    ])
