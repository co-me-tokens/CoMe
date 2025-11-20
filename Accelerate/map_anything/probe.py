import torch

import typing as T
from Network.map_anything import MapAnything


ProbeLocation: T.TypeAlias = tuple[T.Literal["DINO"], int]

def get_block(model: MapAnything, loc: ProbeLocation) -> torch.nn.Module:
    match loc:
        case ("DINO", layer_cnt):
            return model.encoder.model.blocks[layer_cnt]
    raise ValueError(f"Unknown probe location: {loc}")

def set_block(model: MapAnything, loc: ProbeLocation, new_block: torch.nn.Module) -> None:
    match loc:
        case ("DINO", layer_cnt):
            model.encoder.model.blocks[layer_cnt] = new_block        
        case _:
            raise ValueError(f"Unknown probe location: {loc}")
