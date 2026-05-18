"""Thin profiling harness for ncu / nsys.

Usage:
    nsys profile python -m test.auto_test.profile_token_merge --config medium
    ncu  --set detailed --kernel-name token_merge_scatter_kernel_vec4 \
         python -m test.auto_test.profile_token_merge --config medium
"""

import argparse
import torch

from src.accelerate.token_merger.co_me import func_token_merge, func_token_split

CONFIGS = {
    "small":  dict(B=1,   M=256,  G=4, S=0, C=512),
    "medium": dict(B=32,  M=1024, G=4, S=2, C=1024),
    "large":  dict(B=128, M=2048, G=4, S=2, C=2048),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS.keys(), default="medium")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    B, M, G, S, C = cfg["B"], cfg["M"], cfg["G"], cfg["S"], cfg["C"]
    N = S + M * G

    tokens = torch.randn(B, N, C, device="cuda")
    mask = torch.rand(B, M, device="cuda") < 0.5

    for _ in range(args.warmup):
        func_token_merge(tokens, S, G, mask)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        torch.cuda.nvtx.range_push("token_merge_cuda")
        jagged, rev_id = func_token_merge(tokens, S, G, mask)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("token_split_cuda")
        func_token_split(jagged, rev_id, (1, B, N, C))
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
