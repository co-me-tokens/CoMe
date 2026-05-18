"""CUDA token merge / split correctness tests.

Tests the CUDA kernels by comparing their output against the pure-PyTorch
reference implementation across hand-crafted and randomised inputs.
"""

import pytest
import torch

from src.accelerate.token_merger.co_me import (
    func_token_merge,
    func_token_split,
)
from src.utility.feature import CUDAExtension
from src.interface.token_merger import ITokenMerger


DEVICE = "cuda"


# ==== Reuse the hand-crafted cases from unit_test on CUDA ====

from test.unit_test.test_token_merge_func import (
    manual_designed_cases,
    ai_generated_test_cases,
    Datapoint,
)


def _to_cuda(case: Datapoint) -> Datapoint:
    return Datapoint(
        input_tokens=case.input_tokens.cuda(),
        input_merge_mask=case.input_merge_mask.cuda(),
        input_start_index=case.input_start_index,
        input_group_size=case.input_group_size,
        output_tokens=case.output_tokens.cuda(),
        output_offset=case.output_offset.cuda(),
        output_revidx=case.output_revidx.cuda(),
        output_weight=case.output_weight.cuda(),
        restore_tokens=case.restore_tokens.cuda(),
    )


@pytest.mark.parametrize("case", [
    _to_cuda(c) for c in manual_designed_cases + ai_generated_test_cases
])
def test_cuda_merge_matches_expected(case: Datapoint):
    merged, rev_id = func_token_merge(
        tokens=case.input_tokens,
        merge_mask=case.input_merge_mask,
        start_index=case.input_start_index,
        group_size=case.input_group_size,
    )
    torch.testing.assert_close(merged.tokens, case.output_tokens)
    torch.testing.assert_close(merged.offset, case.output_offset)
    torch.testing.assert_close(merged.weight, case.output_weight)
    torch.testing.assert_close(rev_id, case.output_revidx)


@pytest.mark.parametrize("case", [
    _to_cuda(c) for c in manual_designed_cases + ai_generated_test_cases
])
def test_cuda_split_matches_expected(case: Datapoint):
    merged, rev_id = func_token_merge(
        tokens=case.input_tokens,
        merge_mask=case.input_merge_mask,
        start_index=case.input_start_index,
        group_size=case.input_group_size,
    )
    s, n, c = case.input_tokens.shape
    restored = func_token_split(merged, rev_id, (1, s, n, c))
    torch.testing.assert_close(restored, case.restore_tokens)


# ==== Randomised tests: CUDA vs PyTorch reference ====

_RANDOM_CONFIGS = [
    # (B, M, G, start_index, C)
    (1,  4,   2, 0,  8),
    (2,  8,   2, 1,  16),
    (4,  16,  4, 2,  32),
    (8,  32,  4, 0,  64),
    (2,  64,  8, 4,  128),
    (16, 128, 2, 0,  256),
    (1,  256, 4, 1,  512),
    (4,  512, 2, 0,  1024),
    (2,  1024, 4, 2, 512),
]


@pytest.mark.parametrize("B,M,G,S,C", _RANDOM_CONFIGS)
def test_cuda_vs_pytorch_merge(B: int, M: int, G: int, S: int, C: int):
    N = S + M * G
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_ratio = torch.rand(1).item() * 0.8 + 0.1
    merge_mask = torch.rand(B, M, device=DEVICE) < merge_ratio

    with CUDAExtension.disable():
        ref_jagged, ref_rev = func_token_merge(tokens, S, G, merge_mask)
    cuda_jagged, cuda_rev = func_token_merge(tokens, S, G, merge_mask)

    torch.testing.assert_close(cuda_jagged.tokens, ref_jagged.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cuda_jagged.offset, ref_jagged.offset)
    torch.testing.assert_close(cuda_jagged.weight, ref_jagged.weight)
    torch.testing.assert_close(cuda_rev, ref_rev)


@pytest.mark.parametrize("B,M,G,S,C", _RANDOM_CONFIGS)
def test_cuda_vs_pytorch_roundtrip(B: int, M: int, G: int, S: int, C: int):
    N = S + M * G
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.rand(B, M, device=DEVICE) < 0.5

    cuda_jagged, cuda_rev = func_token_merge(tokens, S, G, merge_mask)
    restored = func_token_split(cuda_jagged, cuda_rev, (1, B, N, C))

    with CUDAExtension.disable():
        ref_jagged, ref_rev = func_token_merge(tokens, S, G, merge_mask)
        ref_restored = func_token_split(ref_jagged, ref_rev, (1, B, N, C))

    torch.testing.assert_close(restored, ref_restored, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mask_val", [True, False])
def test_cuda_uniform_mask(mask_val: bool):
    B, M, G, S, C = 4, 32, 4, 2, 64
    N = S + M * G
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.full((B, M), mask_val, dtype=torch.bool, device=DEVICE)

    with CUDAExtension.disable():
        ref_jagged, ref_rev = func_token_merge(tokens, S, G, merge_mask)
    cuda_jagged, cuda_rev = func_token_merge(tokens, S, G, merge_mask)

    torch.testing.assert_close(cuda_jagged.tokens, ref_jagged.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cuda_jagged.offset, ref_jagged.offset)
    torch.testing.assert_close(cuda_jagged.weight, ref_jagged.weight)
    torch.testing.assert_close(cuda_rev, ref_rev)


def test_cuda_group_size_1():
    B, M, S, C = 2, 16, 1, 32
    G = 1
    N = S + M * G
    tokens = torch.randn(B, N, C, device=DEVICE)
    merge_mask = torch.rand(B, M, device=DEVICE) < 0.5

    with CUDAExtension.disable():
        ref_jagged, ref_rev = func_token_merge(tokens, S, G, merge_mask)
    cuda_jagged, cuda_rev = func_token_merge(tokens, S, G, merge_mask)

    torch.testing.assert_close(cuda_jagged.tokens, ref_jagged.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cuda_jagged.offset, ref_jagged.offset)
    torch.testing.assert_close(cuda_jagged.weight, ref_jagged.weight)
    torch.testing.assert_close(cuda_rev, ref_rev)
