from ..vggt import get_VGGT, VGGT
from .implement import enforce_flashattn_vggt


def get_VGGT_star() -> VGGT:
    return enforce_flashattn_vggt(get_VGGT())
