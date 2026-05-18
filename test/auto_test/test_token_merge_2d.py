"""2D token merge / split correctness tests.

Tests the naive pure-PyTorch and the CUDA-accelerated 2D merge
implementations against hand-crafted expected outputs and each other.
"""

import pytest
import torch

import jaxtyping as jt
from dataclasses import dataclass

from src.accelerate.token_merger.co_me_2d import (
    naive_func_token_merge_2d,
    func_token_merge_2d,
)
from src.accelerate.token_merger.co_me import func_token_split
from src.utility.feature import CUDAExtension


DEVICE = "cuda"


# ==== Datapoint ====

@dataclass(kw_only=True)
class Datapoint2D:
    input_tokens:      jt.Float[torch.Tensor, "B N C"]
    input_merge_mask:  jt.Bool[torch.Tensor, "B M"]
    input_start_index: int
    input_group_h:     int
    input_group_w:     int
    input_H:           int
    input_W:           int

    output_tokens:  jt.Float[torch.Tensor, "1 P C"]
    output_offset:  jt.Int[torch.Tensor, "B+1"]
    output_revidx:  jt.Int[torch.Tensor, "B*N"]
    output_weight:  jt.Float[torch.Tensor, "P"]

    restore_tokens: jt.Float[torch.Tensor, "B N C"]


# ==== Hand-crafted test cases ====

manual_designed_cases: list[Datapoint2D] = [
    # Case 1: B=1, H=4, W=4, group_h=2, group_w=2, start_index=0
    # Blocks: (0,0)={0,1,4,5} (0,1)={2,3,6,7} (1,0)={8,9,12,13} (1,1)={10,11,14,15}
    # mask=[True, False, True, False] → merge blocks 0,2; keep blocks 1,3
    Datapoint2D(
        input_tokens=torch.arange(16, dtype=torch.float).unsqueeze(0).unsqueeze(-1),
        input_merge_mask=torch.tensor([[True, False, True, False]]),
        input_start_index=0,
        input_group_h=2,
        input_group_w=2,
        input_H=4,
        input_W=4,
        output_tokens=torch.tensor([[2.5, 2., 3., 6., 7., 10.5, 10., 11., 14., 15.]]).unsqueeze(-1),
        output_offset=torch.tensor([0, 10], dtype=torch.long),
        output_revidx=torch.tensor(
            [[0, 0, 1, 2, 0, 0, 3, 4, 5, 5, 6, 7, 5, 5, 8, 9]], dtype=torch.long
        ).flatten(),
        output_weight=torch.tensor([4., 1., 1., 1., 1., 4., 1., 1., 1., 1.]),
        restore_tokens=torch.tensor(
            [[2.5, 2.5, 2., 3., 2.5, 2.5, 6., 7., 10.5, 10.5, 10., 11., 10.5, 10.5, 14., 15.]]
        ).unsqueeze(-1),
    ),
    # Case 2: B=2, H=2, W=4, group_h=2, group_w=2, start_index=1
    # Batch 0: s=99, img=0..7, mask=[True, False]
    # Batch 1: s=100, img=8..15, mask=[False, True]
    Datapoint2D(
        input_tokens=torch.tensor([
            [99., 0., 1., 2., 3., 4., 5., 6., 7.],
            [100., 8., 9., 10., 11., 12., 13., 14., 15.],
        ]).unsqueeze(-1),
        input_merge_mask=torch.tensor([
            [True, False],
            [False, True],
        ]),
        input_start_index=1,
        input_group_h=2,
        input_group_w=2,
        input_H=2,
        input_W=4,
        output_tokens=torch.tensor(
            [[99., 2.5, 2., 3., 6., 7., 100., 8., 9., 12., 13., 12.5]]
        ).unsqueeze(-1),
        output_offset=torch.tensor([0, 6, 12], dtype=torch.long),
        output_revidx=torch.tensor(
            [0, 1, 1, 2, 3, 1, 1, 4, 5, 6, 7, 8, 11, 11, 9, 10, 11, 11],
            dtype=torch.long,
        ),
        output_weight=torch.tensor([1., 4., 1., 1., 1., 1., 1., 1., 1., 1., 1., 4.]),
        restore_tokens=torch.tensor([
            [99., 2.5, 2.5, 2., 3., 2.5, 2.5, 6., 7.],
            [100., 8., 9., 12.5, 12.5, 12., 13., 12.5, 12.5],
        ]).unsqueeze(-1),
    ),
    # Case 3: B=1, H=4, W=6, group_h=2, group_w=3, start_index=0, all merged
    # 4 blocks of 6 tokens each
    Datapoint2D(
        input_tokens=torch.arange(24, dtype=torch.float).unsqueeze(0).unsqueeze(-1),
        input_merge_mask=torch.tensor([[True, True, True, True]]),
        input_start_index=0,
        input_group_h=2,
        input_group_w=3,
        input_H=4,
        input_W=6,
        output_tokens=torch.tensor([[4., 7., 16., 19.]]).unsqueeze(-1),
        output_offset=torch.tensor([0, 4], dtype=torch.long),
        output_revidx=torch.tensor(
            [[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3]],
            dtype=torch.long,
        ).flatten(),
        output_weight=torch.tensor([6., 6., 6., 6.]),
        restore_tokens=torch.tensor(
            [[4., 4., 4., 7., 7., 7., 4., 4., 4., 7., 7., 7.,
              16., 16., 16., 19., 19., 19., 16., 16., 16., 19., 19., 19.]]
        ).unsqueeze(-1),
    ),
]


def _to_cuda(case: Datapoint2D) -> Datapoint2D:
    return Datapoint2D(
        input_tokens=case.input_tokens.cuda(),
        input_merge_mask=case.input_merge_mask.cuda(),
        input_start_index=case.input_start_index,
        input_group_h=case.input_group_h,
        input_group_w=case.input_group_w,
        input_H=case.input_H,
        input_W=case.input_W,
        output_tokens=case.output_tokens.cuda(),
        output_offset=case.output_offset.cuda(),
        output_revidx=case.output_revidx.cuda(),
        output_weight=case.output_weight.cuda(),
        restore_tokens=case.restore_tokens.cuda(),
    )


CUDA_TEST_CASES = [_to_cuda(c) for c in manual_designed_cases]


# ==== Naive merge against expected ====

@pytest.mark.parametrize("case", CUDA_TEST_CASES)
def test_naive_merge_2d(case: Datapoint2D):
    merged, rev_id = naive_func_token_merge_2d(
        tokens=case.input_tokens,
        start_index=case.input_start_index,
        group_h=case.input_group_h,
        group_w=case.input_group_w,
        H=case.input_H,
        W=case.input_W,
        merge_mask=case.input_merge_mask,
    )
    torch.testing.assert_close(merged.tokens, case.output_tokens)
    torch.testing.assert_close(merged.offset, case.output_offset)
    torch.testing.assert_close(merged.weight, case.output_weight)
    torch.testing.assert_close(rev_id, case.output_revidx)


# ==== Naive roundtrip ====

@pytest.mark.parametrize("case", CUDA_TEST_CASES)
def test_naive_roundtrip_2d(case: Datapoint2D):
    merged, rev_id = naive_func_token_merge_2d(
        tokens=case.input_tokens,
        start_index=case.input_start_index,
        group_h=case.input_group_h,
        group_w=case.input_group_w,
        H=case.input_H,
        W=case.input_W,
        merge_mask=case.input_merge_mask,
    )
    B, N, C = case.input_tokens.shape
    restored = func_token_split(merged, rev_id, (1, B, N, C))
    torch.testing.assert_close(restored, case.restore_tokens)


# ==== CUDA-accelerated merge matches naive ====

@pytest.mark.parametrize("case", CUDA_TEST_CASES)
def test_cuda_merge_2d_matches_expected(case: Datapoint2D):
    merged, rev_id = func_token_merge_2d(
        tokens=case.input_tokens,
        start_index=case.input_start_index,
        group_h=case.input_group_h,
        group_w=case.input_group_w,
        H=case.input_H,
        W=case.input_W,
        merge_mask=case.input_merge_mask,
    )
    torch.testing.assert_close(merged.tokens, case.output_tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(merged.offset, case.output_offset)
    torch.testing.assert_close(merged.weight, case.output_weight)
    torch.testing.assert_close(rev_id, case.output_revidx)


@pytest.mark.parametrize("case", CUDA_TEST_CASES)
def test_cuda_roundtrip_2d(case: Datapoint2D):
    merged, rev_id = func_token_merge_2d(
        tokens=case.input_tokens,
        start_index=case.input_start_index,
        group_h=case.input_group_h,
        group_w=case.input_group_w,
        H=case.input_H,
        W=case.input_W,
        merge_mask=case.input_merge_mask,
    )
    B, N, C = case.input_tokens.shape
    restored = func_token_split(merged, rev_id, (1, B, N, C))
    torch.testing.assert_close(restored, case.restore_tokens, atol=1e-5, rtol=1e-5)


# ==== Randomised parity: CUDA vs naive ====

_RANDOM_CONFIGS = [
    # (B, H, W, group_h, group_w, start_index, C)
    (1,  4,  4,  2, 2, 0, 8),
    (2,  4,  6,  2, 3, 1, 16),
    (4,  8,  8,  4, 4, 0, 32),
    (2,  6,  6,  2, 3, 2, 64),
    (8,  8, 16,  4, 4, 0, 128),
    (1, 16, 16,  4, 8, 1, 256),
    (4,  8, 12,  2, 4, 0, 64),
    (2, 12, 12,  4, 6, 3, 32),
]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


@pytest.mark.parametrize("B,H,W,gh,gw,S,C", _RANDOM_CONFIGS)
def test_cuda_vs_naive_merge_2d(B: int, H: int, W: int, gh: int, gw: int, S: int, C: int):
    N = S + H * W
    M = (H // gh) * (W // gw)
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.rand(B, M, device=DEVICE) < 0.5

    naive_jagged, naive_rev = naive_func_token_merge_2d(tokens, S, gh, gw, H, W, merge_mask)
    cuda_jagged, cuda_rev = func_token_merge_2d(tokens, S, gh, gw, H, W, merge_mask)

    torch.testing.assert_close(cuda_jagged.tokens, naive_jagged.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cuda_jagged.offset, naive_jagged.offset)
    torch.testing.assert_close(cuda_jagged.weight, naive_jagged.weight)
    torch.testing.assert_close(cuda_rev, naive_rev)


@pytest.mark.parametrize("B,H,W,gh,gw,S,C", _RANDOM_CONFIGS)
def test_cuda_vs_naive_roundtrip_2d(B: int, H: int, W: int, gh: int, gw: int, S: int, C: int):
    N = S + H * W
    M = (H // gh) * (W // gw)
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.rand(B, M, device=DEVICE) < 0.5

    naive_jagged, naive_rev = naive_func_token_merge_2d(tokens, S, gh, gw, H, W, merge_mask)
    cuda_jagged, cuda_rev = func_token_merge_2d(tokens, S, gh, gw, H, W, merge_mask)

    naive_restored = func_token_split(naive_jagged, naive_rev, (1, B, N, C))
    cuda_restored = func_token_split(cuda_jagged, cuda_rev, (1, B, N, C))

    torch.testing.assert_close(cuda_restored, naive_restored, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mask_val", [True, False])
def test_uniform_mask_2d(mask_val: bool):
    B, H, W, gh, gw, S, C = 4, 8, 8, 4, 4, 2, 64
    N = S + H * W
    M = (H // gh) * (W // gw)
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.full((B, M), mask_val, dtype=torch.bool, device=DEVICE)

    naive_jagged, naive_rev = naive_func_token_merge_2d(tokens, S, gh, gw, H, W, merge_mask)
    cuda_jagged, cuda_rev = func_token_merge_2d(tokens, S, gh, gw, H, W, merge_mask)

    torch.testing.assert_close(cuda_jagged.tokens, naive_jagged.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cuda_jagged.offset, naive_jagged.offset)
    torch.testing.assert_close(cuda_jagged.weight, naive_jagged.weight)
    torch.testing.assert_close(cuda_rev, naive_rev)


# ==== Validation ====

@pytest.mark.parametrize(
    "tokens, mask, start, gh, gw, H, W, expected_msg",
    [
        # H not divisible by group_h
        (torch.randn(1, 15, 4), torch.ones(1, 3, dtype=torch.bool), 0, 2, 3, 3, 5, "H.*divisible"),
        # W not divisible by group_w
        (torch.randn(1, 12, 4), torch.ones(1, 2, dtype=torch.bool), 0, 2, 5, 4, 3, "W.*divisible"),
        # merge_mask batch mismatch
        (torch.randn(2, 16, 4), torch.ones(3, 4, dtype=torch.bool), 0, 2, 2, 4, 4, "merge_mask batch"),
        # N != start_index + H * W
        (torch.randn(1, 10, 4), torch.ones(1, 4, dtype=torch.bool), 0, 2, 2, 4, 4, "Expected N"),
        # M mismatch
        (torch.randn(1, 16, 4), torch.ones(1, 2, dtype=torch.bool), 0, 2, 2, 4, 4, "Expected M"),
        # start_index < 0
        (torch.randn(1, 16, 4), torch.ones(1, 4, dtype=torch.bool), -1, 2, 2, 4, 4, "start_index"),
        # group_h <= 0
        (torch.randn(1, 16, 4), torch.ones(1, 4, dtype=torch.bool), 0, 0, 2, 4, 4, "group_h.*group_w.*> 0"),
        # merge_mask not bool
        (torch.randn(1, 16, 4), torch.ones(1, 4, dtype=torch.float32), 0, 2, 2, 4, 4, "torch.bool"),
    ],
)
def test_2d_merge_validation(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    gh: int,
    gw: int,
    H: int,
    W: int,
    expected_msg: str,
):
    with pytest.raises(ValueError, match=expected_msg):
        naive_func_token_merge_2d(tokens=tokens, merge_mask=mask, start_index=start,
                                  group_h=gh, group_w=gw, H=H, W=W)
