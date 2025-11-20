import torch
import random
from pathlib import Path
from rich.progress import track


from ..Interface import DepthDataset, DepthData
from .dataset import TartanAirGTDepthDataset, TartanAirMonocularDataset


class TartanAir_DepthDataset(DepthDataset):
    def __init__(self, root: Path | str, sequence_length: int) -> None:
        self.root, self.sequence_length = root, sequence_length
        
        all_sequences = self._list_all_trajectories(Path(root))
        self.all_images = [
            TartanAirMonocularDataset(Path(seq, "image_left"))
            for seq in track(all_sequences, description="Loading all TartanAir images", transient=True)
        ]
        self.all_depths = [
            TartanAirGTDepthDataset(Path(seq, "depth_left"), compressed=True)
            for seq in track(all_sequences, description="Loading all TartanAir depths", transient=True)
        ]
        assert (len(i) == len(d) for i, d in zip(self.all_images, self.all_depths))
        
        self.all_samples = [
            (seq_idx, sample_range)
            for seq_idx, sequence in enumerate(self.all_images)
            for sample_range in self._low_variance_sampling(len(sequence), sequence_length)
        ]
        self.length = len(self.all_samples)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> DepthData:
        seq_idx, sample_range = self.all_samples[index]
        images = torch.stack([self.all_images[seq_idx][idx] for idx in range(*sample_range)], dim=1)
        depths = torch.stack([self.all_depths[seq_idx][idx] for idx in range(*sample_range)], dim=1)
        return DepthData(images=images, gt_depths=depths)

    # Helper functions
    @staticmethod
    def _low_variance_sampling(total_length: int, segment_length: int) -> list[tuple[int, int, int]]:
        if segment_length > total_length: return []
        start_point_range = (0, min(segment_length-1, total_length-segment_length))
        start_point       = random.randint(*start_point_range)
        
        return [
            (s, s + segment_length, 1)
            for s in range(start_point, total_length, segment_length)
            if (s + segment_length) < total_length
        ]
    
    @staticmethod
    def _list_all_trajectories(root: Path) -> list[Path]:
        envs = [d for d in root.iterdir() if d.is_dir()]
        seqs = [
            s 
            for env in envs if (Path(env, "Data_fast").exists())
            for s in Path(env, "Data_fast").iterdir() if (s.is_dir() and (s.name.startswith("P")))
        ]
        return seqs
