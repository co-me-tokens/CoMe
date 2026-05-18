from typing import Literal

from .api import DepthAnything3


T_DepthAnything3Type = Literal["da3-base", "da3-giant", "da3-large", "da3-small"]

_MODEL_REPO_IDS: dict[T_DepthAnything3Type, str] = {
    "da3-giant": "depth-anything/DA3-GIANT-1.1",
    "da3-large": "depth-anything/DA3-LARGE-1.1",
    "da3-base" : "depth-anything/DA3-BASE",
    "da3-small": "depth-anything/DA3-SMALL",
}


def get_DepthAnything3(model_type: T_DepthAnything3Type = "da3-giant") -> DepthAnything3:
    repo_id = _MODEL_REPO_IDS[model_type]    
    return DepthAnything3.from_pretrained(repo_id)
