import torch

import typing as T
from Network.vggt import VGGT



ProbeLocation: T.TypeAlias = tuple[T.Literal["DINO"], int] | tuple[T.Literal["frame_block", "global_block"], int]

def get_block(model: VGGT, loc: ProbeLocation) -> torch.nn.Module:
    match loc:
        case ("DINO", layer_cnt):
            return model.aggregator.patch_embed.blocks[layer_cnt]
        case ("frame_block", layer_cnt):
            return model.aggregator.frame_blocks[layer_cnt]
        case ("global_block", layer_cnt):
            return model.aggregator.global_blocks[layer_cnt]
    raise ValueError(f"Unknown probe location: {loc}")

def set_block(model: VGGT, loc: ProbeLocation, new_block: torch.nn.Module) -> None:
    match loc:
        case ("DINO", layer_cnt):
            model.aggregator.patch_embed.blocks[layer_cnt] = new_block
        case ("frame_block", layer_cnt):
            model.aggregator.frame_blocks[layer_cnt] = new_block
        case ("global_block", layer_cnt):
            model.aggregator.global_blocks[layer_cnt] = new_block
        case _:
            raise ValueError(f"Unknown probe location: {loc}")
