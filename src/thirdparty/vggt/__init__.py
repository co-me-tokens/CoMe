import torch as _torch
from .models.vggt import VGGT


def get_VGGT() -> VGGT:
    URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model = VGGT()
    state_dict = _torch.hub.load_state_dict_from_url(URL)
    incompatible = model.load_state_dict(state_dict, strict=False)

    unexpected_non_track = [k for k in incompatible.unexpected_keys if not k.startswith("track_head.")]
    if unexpected_non_track:
        raise RuntimeError(
            "Unexpected checkpoint keys not related to removed track head: "
            f"{unexpected_non_track}"
        )

    if incompatible.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required model keys: "
            f"{incompatible.missing_keys}"
        )
    return model
