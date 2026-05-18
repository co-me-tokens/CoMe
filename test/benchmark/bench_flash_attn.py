"""Benchmark flash_attn_varlen_qkvpacked_func: original vs local CUDA reproduction."""

import torch
import torch.utils.benchmark as benchmark

from flash_attn import flash_attn_varlen_qkvpacked_func as orig_fn
from src.cuda_extension.flash_attn import flash_attn_varlen_qkvpacked_func as ours_fn

# (label, batch, seqlen, nheads, headdim, causal, dtype)
CONFIGS = [
    ("tiny",           4,   128,   8,  64,  False, torch.float16),
    ("small",          8,   256,  16,  64,  False, torch.float16),
    ("med",           16,   512,  16, 128,  False, torch.float16),
    ("large",         32,  1024,  32, 128,  False, torch.float16),
    ("xlarge",        64,  2048,  32, 128,  False, torch.float16),
    ("small-causal",   8,   256,  16,  64,  True,  torch.float16),
    ("med-causal",    16,   512,  16, 128,  True,  torch.float16),
    ("large-causal",  32,  1024,  32, 128,  True,  torch.float16),
    ("small-bf16",     8,   256,  16,  64,  False, torch.bfloat16),
    ("med-bf16",      16,   512,  16, 128,  False, torch.bfloat16),
    ("large-bf16",    32,  1024,  32, 128,  False, torch.bfloat16),
    ("large-c-bf16",  32,  1024,  32, 128,  True,  torch.bfloat16),
]

DEVICE = "cuda"


def _make_inputs(batch: int, seqlen: int, nheads: int, headdim: int, dtype: torch.dtype):
    total = batch * seqlen
    qkv = torch.randn(total, 3, nheads, headdim, device=DEVICE, dtype=dtype)
    cu_seqlens = torch.arange(0, batch + 1, device=DEVICE, dtype=torch.int32) * seqlen
    return qkv, cu_seqlens, seqlen


def main():
    print("=" * 90)
    print("FLASH ATTENTION VARLEN QKV-PACKED FORWARD BENCHMARK")
    print("=" * 90)

    results: list[benchmark.Measurement] = []

    for label, batch, seqlen, nheads, headdim, causal, dtype in CONFIGS:
        qkv, cu_seqlens, max_seqlen = _make_inputs(batch, seqlen, nheads, headdim, dtype)
        total = batch * seqlen
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        sub = f"{label} b={batch} s={seqlen} h={nheads} d={headdim} {'causal' if causal else 'full'} {dtype_str}"

        t_orig = benchmark.Timer(
            stmt="fn(qkv, cu, ms, causal=causal)",
            globals={"fn": orig_fn, "qkv": qkv, "cu": cu_seqlens,
                     "ms": max_seqlen, "causal": causal},
            label="flash_attn_varlen_qkvpacked",
            sub_label=sub,
            description="Original",
        )
        t_ours = benchmark.Timer(
            stmt="fn(qkv, cu, ms, causal=causal)",
            globals={"fn": ours_fn, "qkv": qkv, "cu": cu_seqlens,
                     "ms": max_seqlen, "causal": causal},
            label="flash_attn_varlen_qkvpacked",
            sub_label=sub,
            description="Ours",
        )

        r_orig = t_orig.blocked_autorange(min_run_time=2.0)
        r_ours = t_ours.blocked_autorange(min_run_time=2.0)
        results.extend([r_orig, r_ours])

        ratio = r_orig.median / r_ours.median
        print(f"\n{label:16s}  total={total:<6d}  h={nheads:<3d}  d={headdim:<4d}  "
              f"{'causal' if causal else 'full':6s}  {dtype_str}")
        print(f"  Original : {r_orig.median*1e3:8.3f} ms")
        print(f"  Ours     : {r_ours.median*1e3:8.3f} ms")
        print(f"  Ratio    : {ratio:8.4f}x")

    print("\n" + "=" * 90)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
