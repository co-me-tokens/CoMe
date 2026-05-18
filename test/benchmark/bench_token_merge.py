"""Benchmark token_merge and token_split: PyTorch vs CUDA kernel."""

import torch
import torch.utils.benchmark as benchmark

from src.accelerate.token_merger.co_me import (
    func_token_merge,
    func_token_split,
)
from src.utility.feature import CUDAExtension


def _func_token_merge_py(*args, **kwargs):
    with CUDAExtension.disable():
        return func_token_merge(*args, **kwargs)


CONFIGS = [
    # label,           B,   M,    G,  S,   C
    ("Small",          1,   256,  4,  0,   512),
    ("Medium",         32,  1024, 4,  2,   1024),
    ("Large",          128, 2048, 4,  2,   2048),
]

DEVICE = "cuda"


def _make_inputs(B, M, G, S, C):
    N = S + M * G
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.rand(B, M, device=DEVICE) < 0.5
    return tokens, merge_mask


def bench_merge():
    print("=" * 80)
    print("TOKEN MERGE BENCHMARK")
    print("=" * 80)
    results = []
    for label, B, M, G, S, C in CONFIGS:
        tokens, mask = _make_inputs(B, M, G, S, C)
        N = S + M * G

        t_py = benchmark.Timer(
            stmt="_func_token_merge_py(tokens, S, G, mask)",
            globals={"_func_token_merge_py": _func_token_merge_py,
                     "tokens": tokens, "mask": mask, "S": S, "G": G},
            label="token_merge",
            sub_label=f"{label} B={B} N={N} C={C} G={G}",
            description="PyTorch",
        )
        t_cu = benchmark.Timer(
            stmt="func_token_merge(tokens, S, G, mask)",
            globals={"func_token_merge": func_token_merge,
                     "tokens": tokens, "mask": mask, "S": S, "G": G},
            label="token_merge",
            sub_label=f"{label} B={B} N={N} C={C} G={G}",
            description="CUDA",
        )

        r_py = t_py.blocked_autorange(min_run_time=2.0)
        r_cu = t_cu.blocked_autorange(min_run_time=2.0)
        results.extend([r_py, r_cu])

        speedup = r_py.median / r_cu.median
        print(f"\n{label:8s}  B={B:<4d} N={N:<5d} C={C:<5d} G={G}")
        print(f"  PyTorch : {r_py.median*1e3:8.3f} ms")
        print(f"  CUDA    : {r_cu.median*1e3:8.3f} ms")
        print(f"  Speedup : {speedup:8.2f}x")

    print("\n" + "=" * 80)
    compare = benchmark.Compare(results)
    compare.print()


def bench_split():
    print("\n" + "=" * 80)
    print("TOKEN SPLIT BENCHMARK")
    print("=" * 80)
    results = []
    for label, B, M, G, S, C in CONFIGS:
        tokens, mask = _make_inputs(B, M, G, S, C)
        N = S + M * G

        jagged_py, rev_py = _func_token_merge_py(tokens, S, G, mask)
        jagged_cu, rev_cu = func_token_merge(tokens, S, G, mask)
        bsnc = (1, B, N, C)

        t_py = benchmark.Timer(
            stmt="flattened[rev_id].reshape(BN, N, C)",
            globals={"flattened": jagged_py.tokens.squeeze(0), "rev_id": rev_py,
                     "BN": B, "N": N, "C": C},
            label="token_split",
            sub_label=f"{label} B={B} N={N} C={C} G={G}",
            description="PyTorch",
        )
        t_cu = benchmark.Timer(
            stmt="func_token_split(jagged, rev_id, bsnc)",
            globals={"func_token_split": func_token_split, "jagged": jagged_cu,
                     "rev_id": rev_cu, "bsnc": bsnc},
            label="token_split",
            sub_label=f"{label} B={B} N={N} C={C} G={G}",
            description="CUDA",
        )

        r_py = t_py.blocked_autorange(min_run_time=2.0)
        r_cu = t_cu.blocked_autorange(min_run_time=2.0)
        results.extend([r_py, r_cu])

        speedup = r_py.median / r_cu.median
        print(f"\n{label:8s}  B={B:<4d} N={N:<5d} C={C:<5d} G={G}")
        print(f"  PyTorch : {r_py.median*1e3:8.3f} ms")
        print(f"  CUDA    : {r_cu.median*1e3:8.3f} ms")
        print(f"  Speedup : {speedup:8.2f}x")

    print("\n" + "=" * 80)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    bench_merge()
    bench_split()
