import torch
import pypose as pp
from dataclasses import dataclass
import cv2
from pathlib import Path

@dataclass
class DepthData:
    images   : torch.Tensor # B x S x 3 x H x W
    gt_depths: torch.Tensor # B x S x 1 x H x W


@dataclass
class PointData:
    images   : torch.Tensor # B x S x 3 x H x W
    points   : list[torch.Tensor] # [{N1, N2, ..., NB} x 3]

@dataclass
class PoseData:
    images   : torch.Tensor # B x S x 3 x H x W
    poses    : pp.LieTensor # B x S x 7, pp.SE3 LieTensor, OpenCV coordinate convention / EDN convention
    intrinsic: torch.Tensor | None = None# B x S x 3 x 3
    extra: dict | None = None

@dataclass 
class MVSData: 
    images    : torch.Tensor # B x S x 3 x H x W 
    gt_depths : torch.Tensor # B x S x 1 x H x W
    points    : list[torch.Tensor] # [{N1, N2, ..., NB} x 3]
    poses     : pp.LieTensor # B x S x 7, pp.SE3 LieTensor, OpenCV coordinate convention / EDN convention
    intrinsics: torch.Tensor # B x S x 3 x 3

class MVSDataset(torch.utils.data.Dataset[MVSData]):
    @staticmethod
    def collate(batch: list[MVSData]) -> MVSData:
        return MVSData(
            images    =torch.cat([b.images     for b in batch], dim=0),
            gt_depths =torch.cat([b.gt_depths  for b in batch], dim=0),
            points    =[b for sample in batch for b in sample.points],
            poses     =pp.SE3(torch.cat([b.poses      for b in batch], dim=0)),
            intrinsics=torch.cat([b.intrinsics for b in batch], dim=0)
        )
class DepthDataset(torch.utils.data.Dataset[DepthData]):
    @staticmethod
    def collate(batch: list[DepthData]) -> DepthData:
        return DepthData(
            images   =torch.cat([b.images    for b in batch], dim=0),
            gt_depths=torch.cat([b.gt_depths for b in batch], dim=0)
        )

class PointDataset(torch.utils.data.Dataset[PointData]):
    @staticmethod
    def collate(batch: list[PointData]) -> PointData:
        return PointData(
            images    =torch.cat([b.images for b in batch], dim=0),
            points    =[b for sample in batch for b in sample.points]
        )

class PoseDataset(torch.utils.data.Dataset[PoseData]):
    @staticmethod
    def collate(batch: list[PoseData]) -> PoseData:
        return PoseData(
            images    =torch.cat([b.images for b in batch], dim=0),
            poses     =pp.SE3(torch.cat([b.poses for b in batch], dim=0))
        )


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        video_path: str | Path, 
        target_size: tuple[int, int] = (378, 504)):
        self.video_path = Path(video_path)
        self.target_size = target_size
        self.frames = self._load_video_frames()


    def _load_video_frames(self) -> list[torch.Tensor]:
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.target_size[::-1])
            frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)
            frames.append(frame_tensor)

        cap.release()
        return frames

    def __len__(self) -> int:
        return max(0, len(self.frames))

    def __getitem__(self, idx) -> torch.Tensor:
        return self.frames[idx]
