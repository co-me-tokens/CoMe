"""CUDA Extension for PyTorch — Custom kernel operations."""
from .flash_attn import (
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func_w_perkey_bias,
    infer_flash_attn_varlen_max_seqlen,
)

__all__ = [
    "co_me_cuext",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_qkvpacked_func_w_perkey_bias",
    "infer_flash_attn_varlen_max_seqlen",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    if name == "co_me_cuext":
        from .co_me import co_me_cuext

        return co_me_cuext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
