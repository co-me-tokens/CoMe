"""DTU MVS dataset loader for multi-view geometry tasks.

Uses the MVSNet_Preprocessed format: preprocessed rectified images (640x512),
camera parameters (extrinsic + intrinsic), and precomputed depth maps (160x128 PFM).
"""

import numpy as np
import torch
import torch.nn.functional as F
import pypose as pp
from pathlib import Path
from PIL import Image
from rich.progress import track

from ..interface.dataset import MultiViewGeometryDataset, MultiViewTask
from ..interface.geometric_model import MultiViewInput, SceneGeometry, PoseConvention

IMG_H, IMG_W = 512, 640


# ==== Module-level helpers ====

def read_pfm(path: Path) -> np.ndarray:
    """Read a single-channel PFM file. Returns an (H, W) float32 array."""
    with open(path, "rb") as f:
        header = f.readline().decode().rstrip()
        if header != "Pf":
            raise ValueError(f"Expected grayscale PFM ('Pf'), got '{header}' in {path}")
        w, h = map(int, f.readline().decode().rstrip().split())
        scale = float(f.readline().decode().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=f"{endian}f").reshape(h, w)
        return np.flipud(data).copy()


def parse_camera_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an MVSNet *_cam.txt file.

    Returns:
        (extrinsic, intrinsic): extrinsic is (4, 4), intrinsic is (3, 3), both float64.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 9:
        raise ValueError(f"Camera file too short ({len(lines)} non-empty lines): {path}")

    extrinsic = np.array([[float(x) for x in lines[i].split()] for i in range(1, 5)])
    intrinsic = np.array([[float(x) for x in lines[i].split()] for i in range(6, 9)])
    return extrinsic, intrinsic


# ==== Dataset ====

class DTU_MVS_Dataset(MultiViewGeometryDataset):
    """Single-scan DTU MVS dataset. Each instance yields exactly one multi-view sample."""

    def __init__(
        self,
        data_root: Path | str,
        scan_id: int,
        segment_length: int,
        lighting: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()

        if not 1 <= segment_length <= 49:
            raise ValueError(f"segment_length must be in [1, 49], got {segment_length}")
        if not 0 <= lighting <= 6:
            raise ValueError(f"lighting must be in [0, 6], got {lighting}")

        data_root = Path(data_root)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {data_root}")

        self.img_dir   = Path(data_root, "Rectified", f"scan{scan_id}_train")
        self.depth_dir = Path(data_root, "Depths",    f"scan{scan_id}_train")
        cam_dir        = Path(data_root, "Cameras",   "train")

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.depth_dir.is_dir():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

        self.segment_length = segment_length
        self.lighting = lighting
        self.device = device

        extrinsics_list: list[np.ndarray] = []
        intrinsics_list: list[np.ndarray] = []
        for v in range(segment_length):
            ext, intr = parse_camera_file(Path(cam_dir, f"{v:08d}_cam.txt"))
            extrinsics_list.append(ext)
            intrinsics_list.append(intr)

        # DTU raw data is in millimeters; convert translation to meters.
        self.extrinsics = np.stack(extrinsics_list)  # (S, 4, 4)
        self.extrinsics[:, :3, 3] *= 1e-3

        # Train intrinsics are at depth resolution (160x128); scale by 4 to image resolution (640x512).
        self.intrinsics = np.stack(intrinsics_list)  # (S, 3, 3)
        self.intrinsics[:, :2, :] *= 4

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> MultiViewTask:
        assert index == 0, f"DTU_MVS_Dataset has only 1 sample, got index {index}"

        S = self.segment_length

        images: list[torch.Tensor] = []
        for v in range(S):
            arr = np.array(Image.open(Path(self.img_dir, f"rect_{v + 1:03d}_{self.lighting}_r5000.png")))
            images.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
        images_t = torch.stack(images).unsqueeze(0)

        depths: list[torch.Tensor] = []
        for v in range(S):
            depths.append(torch.from_numpy(read_pfm(Path(self.depth_dir, f"depth_map_{v:04d}.pfm"))))
        depths_t = torch.stack(depths).unsqueeze(1) * 1e-3  # mm -> meters
        depths_t = F.interpolate(depths_t, size=(IMG_H, IMG_W), mode="nearest").unsqueeze(0)

        w2c = torch.from_numpy(self.extrinsics).float()
        c2w = torch.linalg.inv(w2c)
        world_poses = pp.mat2SE3(c2w)  # type: ignore[operator]
        relative_poses = world_poses[0].Inv() @ world_poses  # type: ignore[attr-defined]
        poses_t = relative_poses.tensor().unsqueeze(0)

        K = torch.from_numpy(self.intrinsics.copy()).float().unsqueeze(0)

        # c2w_rel[v] = w2c[0] @ c2w[v]: all views in first-camera world frame,
        # consistent with the relative poses stored in poses_t.
        c2w_rel = w2c[0] @ c2w                                              # (S, 4, 4)

        v_coords, u_coords = torch.meshgrid(
            torch.arange(IMG_H, dtype=torch.float32),
            torch.arange(IMG_W, dtype=torch.float32),
            indexing="ij",
        )
        pixel_flat = torch.stack([u_coords, v_coords], dim=-1).reshape(-1, 2)  # (H*W, 2)

        pts_world_list: list[torch.Tensor] = []
        for v in range(S):
            depth_v = depths_t[0, v, 0].reshape(-1)                         # (H*W,)
            pts_cam_v = pp.pixel2point(pixel_flat, depth_v, K[0, v])        # (H*W, 3)
            R_v, t_v = c2w_rel[v, :3, :3], c2w_rel[v, :3, 3]
            pts_w = R_v @ pts_cam_v.T + t_v.unsqueeze(-1)                   # (3, H*W)
            pts_world_list.append(pts_w.reshape(3, IMG_H, IMG_W))
        pts_world = torch.stack(pts_world_list)                              # (S, 3, H, W)

        depths_sv = depths_t[0, :, 0]                                       # (S, H, W)

        valid = (depths_sv > 0).unsqueeze(1)                                # (S, 1, H, W)
        points_t = (pts_world * valid).unsqueeze(0)                         # (1, S, 3, H, W)

        mv_input = MultiViewInput(images=images_t, intrinsics=K.clone())
        scene_geo = SceneGeometry(
            pose_convention=(PoseConvention.R, PoseConvention.D, PoseConvention.F),
            depths=depths_t,
            depths_conf=None,
            points=points_t,
            points_conf=None,
            poses=poses_t,
            intrinsics=K,
            infer_mask=None,
        )

        mv_input.to(device=self.device)
        scene_geo.to(device=self.device)

        return MultiViewTask(multiview_input=mv_input, scene_geometry=scene_geo)


# ==== Full dataset ====

DTU_SCAN_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
]


def AllDTU_MVS_Dataset(
    data_root: Path | str,
    segment_length: int,
    lighting: int,
    device: torch.device | str,
) -> torch.utils.data.ConcatDataset[MultiViewTask]:
    return torch.utils.data.ConcatDataset([
        DTU_MVS_Dataset(data_root, scan_id, segment_length, lighting, device)
        for scan_id in track(DTU_SCAN_IDS, description="Loading Full DTU MVS Dataset...")
    ])
