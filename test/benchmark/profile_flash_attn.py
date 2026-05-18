"""Thin profiling harness for ncu / nsys on flash_attn_varlen_qkvpacked_func.

Usage:
    nsys profile python -m test.benchmark.profile_flash_attn --config medium --impl both
    ncu  --set detailed \
         python -m test.benchmark.profile_flash_attn --config large --impl ours
"""

import argparse
from dataclasses import dataclass
from typing import Callable

import torch

from flash_attn import flash_attn_varlen_qkvpacked_func as orig_fn
from src.cuda_extension.flash_attn import flash_attn_varlen_qkvpacked_func as ours_fn


@dataclass(frozen=True)
class _Cfg:
    batch: int
    seqlen: int
    nheads: int
    headdim: int
    causal: bool
    dtype: torch.dtype


CONFIGS: dict[str, _Cfg] = {
    "tiny":   _Cfg(batch=4,  seqlen=128,  nheads=8,  headdim=64,  causal=False, dtype=torch.float16),
    "small":  _Cfg(batch=8,  seqlen=256,  nheads=16, headdim=64,  causal=False, dtype=torch.float16),
    "medium": _Cfg(batch=16, seqlen=512,  nheads=16, headdim=128, causal=False, dtype=torch.float16),
    "large":  _Cfg(batch=32, seqlen=1024, nheads=32, headdim=128, causal=False, dtype=torch.float16),
    "xlarge": _Cfg(batch=64, seqlen=2048, nheads=32, headdim=128, causal=False, dtype=torch.float16),
}


def _make_inputs(batch: int, seqlen: int, nheads: int, headdim: int, dtype: torch.dtype):
    total = batch * seqlen
    qkv = torch.randn(total, 3, nheads, headdim, device="cuda", dtype=dtype)
    cu_seqlens = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * seqlen
    return qkv, cu_seqlens, seqlen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS.keys(), default="medium")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--impl", choices=["ours", "original", "both"], default="both")
    parser.add_argument("--causal", action="store_true", default=None,
                        help="Override causal flag from config")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    causal = args.causal if args.causal is not None else cfg.causal
    qkv, cu_seqlens, max_seqlen = _make_inputs(
        cfg.batch, cfg.seqlen, cfg.nheads, cfg.headdim, cfg.dtype,
    )

    impls: dict[str, Callable[..., object]] = {}
    if args.impl in ("original", "both"):
        impls["original"] = orig_fn
    if args.impl in ("ours", "both"):
        impls["ours"] = ours_fn

    for name, fn in impls.items():
        for _ in range(args.warmup):
            fn(qkv, cu_seqlens, max_seqlen, causal=causal)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        for name, fn in impls.items():
            torch.cuda.nvtx.range_push(f"flash_attn_{name}")
            fn(qkv, cu_seqlens, max_seqlen, causal=causal)
            torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
