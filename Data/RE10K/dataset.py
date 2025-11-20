"""
Taking PixelSplat hosted data and using their decoding method to read out the data.

To download the data, see: http://schadenfreude.csail.mit.edu:8000/
"""

import torch
import json
import pypose as pp
import torchvision.transforms as tvf
from pathlib import Path
from PIL import Image
from io import BytesIO

from ..Interface import PoseDataset, PoseData


class RE10K_PoseDataset(PoseDataset):
    def __init__(self, root: Path | str, sequence_length: int):
        with open(Path(root, "index.json"), "r") as f:
            index_data = json.load(f)
        with open(Path(Path(__file__).parent, "re10k_sequence_lengths.json"), "r") as f:
            length_data = json.load(f)

        self.root = root
        self.sequence_length = sequence_length
        self.kv_pair = [item for item in index_data.items()]
        self.kv_pair.sort(key=lambda item: (item[1], item[0]))
        self.kv_pair = list(filter(lambda kv: length_data[kv[0]] >= self.sequence_length, self.kv_pair))
        
        self.length  = len(self.kv_pair)
        self.chunks  = sorted(list(set(index_data.values())))

    def __len__(self) -> int: 
        return self.length
    
    
    def __getitem__(self, index) -> PoseData:
        print("Warning: This is very slow and Disk-IO bounded! Try to use the 'for sample in dataset' iterator convention to access.")
        chunk   = torch.load(Path(self.root, self.kv_pair[index][1]))
        c_index = [d['key'] for d in chunk].index(self.kv_pair[index][0])
        
        video_data = chunk[c_index]
        intrinsics, pose_c2w = self._decode_pose_data(video_data['cameras'])
        images               = torch.stack([self._decode_tensor_data(d) for d in video_data['images']], dim=0)
        
        return PoseData(
            images=images.unsqueeze(0)[:, :self.sequence_length],
            poses=pp.SE3(pose_c2w[:, :self.sequence_length]),
            intrinsic=intrinsics
        )
    
    @staticmethod
    def _decode_tensor_data(image: torch.Tensor) -> torch.Tensor:
        dec_image = Image.open(BytesIO(image.numpy().tobytes()))
        return tvf.ToTensor()(dec_image)

    @staticmethod
    def _decode_pose_data(pose: torch.Tensor) -> tuple[torch.Tensor, pp.LieTensor]:
        B = pose.size(0)
        
        intrinsics = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(B, 1, 1)
        extrinsics = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(B, 1, 1)
        
        fx, fy, cx, cy = pose[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        
        extrinsics[:, :3] = pose[:, 6:].reshape(B, 3, 4)
        return intrinsics.unsqueeze(0), pp.from_matrix(extrinsics.unsqueeze(0), ltype=pp.SE3_type).Inv()
    
    def __iter__(self):
        def iterative_access():
            data = self
            initial_chunk_id = data.kv_pair[0][1]
            curr_chunk, curr_chunk_name = torch.load(Path(data.root, initial_chunk_id)), initial_chunk_id
            
            for vid, chunk_id in data.kv_pair:
                if chunk_id != curr_chunk_name:
                    print(f"load chunk at {chunk_id}")
                    curr_chunk, curr_chunk_name = torch.load(Path(data.root, chunk_id)), chunk_id
                
                c_index = [d['key'] for d in curr_chunk].index(vid)
        
                video_data = curr_chunk[c_index]
                intrinsics, pose_c2w = self._decode_pose_data(video_data['cameras'])
                if intrinsics.size(1) < data.sequence_length: continue
                
                intrinsics, pose_c2w = intrinsics[:, :data.sequence_length], pose_c2w[:, :data.sequence_length]
                
                images               = torch.stack([self._decode_tensor_data(d) for d in video_data['images'][:data.sequence_length]], dim=0)
                
                yield PoseData(images=images.unsqueeze(0), poses=pp.SE3(pose_c2w), intrinsic=intrinsics, extra={"vid": vid})
        
        return iterative_access()
