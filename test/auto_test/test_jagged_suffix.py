"""JaggedTokens.add_suffix / split_suffix correctness tests.

Tests hand-crafted expected outputs, roundtrip identity, randomised
configs, edge cases, and input validation.
"""

import pytest
import torch
import jaxtyping as jt
from dataclasses import dataclass

from src.interface.token_merger import ITokenMerger

DEVICE = "cuda"


# ==== Datapoint ====

@dataclass(kw_only=True)
class SuffixDatapoint:
    input_tokens: jt.Float[torch.Tensor, "1 P C"]
    input_weight: jt.Float[torch.Tensor, "P"]
    input_offset: jt.Int[torch.Tensor, "M+1"]
    suffix:       jt.Float[torch.Tensor, "M T C"]

    expected_tokens: jt.Float[torch.Tensor, "1 P2 C"]
    expected_weight: jt.Float[torch.Tensor, "P2"]
    expected_offset: jt.Int[torch.Tensor, "M+1"]


# ==== Hand-crafted cases ====

manual_cases: list[SuffixDatapoint] = [
    # Case 1: single segment [a, b, c] + suffix [[x]]
    # Result: [a, b, c, x], offset [0, 4]
    SuffixDatapoint(
        input_tokens=torch.tensor([[[1.], [2.], [3.]]]),
        input_weight=torch.tensor([1., 2., 1.]),
        input_offset=torch.tensor([0, 3]),
        suffix=torch.tensor([[[10.]]]),
        expected_tokens=torch.tensor([[[1.], [2.], [3.], [10.]]]),
        expected_weight=torch.tensor([1., 2., 1., 1.]),
        expected_offset=torch.tensor([0, 4]),
    ),
    # Case 2: two segments [a, b | c] + suffix [[x, y], [z, w]]  (T=2)
    # Seg 0 (len=2): [a, b] -> [a, b, x, y]
    # Seg 1 (len=1): [c]    -> [c, z, w]
    # Result: [a, b, x, y, c, z, w], offset [0, 4, 7]
    SuffixDatapoint(
        input_tokens=torch.tensor([[[1.], [2.], [3.]]]),
        input_weight=torch.tensor([4., 1., 1.]),
        input_offset=torch.tensor([0, 2, 3]),
        suffix=torch.tensor([[[10.], [20.]], [[30.], [40.]]]),
        expected_tokens=torch.tensor([[[1.], [2.], [10.], [20.], [3.], [30.], [40.]]]),
        expected_weight=torch.tensor([4., 1., 1., 1., 1., 1., 1.]),
        expected_offset=torch.tensor([0, 4, 7]),
    ),
    # Case 3: three segments, multi-channel, varying lengths, T=1
    # Seg 0 (len=1): [[1,2]]
    # Seg 1 (len=3): [[3,4],[5,6],[7,8]]
    # Seg 2 (len=2): [[9,10],[11,12]]
    # suffix: [[[100,200]], [[300,400]], [[500,600]]]
    # Result: [[1,2],[100,200], [3,4],[5,6],[7,8],[300,400], [9,10],[11,12],[500,600]]
    # offset: [0, 2, 6, 9]
    SuffixDatapoint(
        input_tokens=torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [11., 12.]]]),
        input_weight=torch.tensor([1., 1., 4., 1., 1., 1.]),
        input_offset=torch.tensor([0, 1, 4, 6]),
        suffix=torch.tensor([[[100., 200.]], [[300., 400.]], [[500., 600.]]]),
        expected_tokens=torch.tensor([[[1., 2.], [100., 200.],
                                       [3., 4.], [5., 6.], [7., 8.], [300., 400.],
                                       [9., 10.], [11., 12.], [500., 600.]]]),
        expected_weight=torch.tensor([1., 1., 1., 4., 1., 1., 1., 1., 1.]),
        expected_offset=torch.tensor([0, 2, 6, 9]),
    ),
]


def _to_cuda(dp: SuffixDatapoint) -> SuffixDatapoint:
    return SuffixDatapoint(
        input_tokens=dp.input_tokens.cuda(),
        input_weight=dp.input_weight.cuda(),
        input_offset=dp.input_offset.cuda(),
        suffix=dp.suffix.cuda(),
        expected_tokens=dp.expected_tokens.cuda(),
        expected_weight=dp.expected_weight.cuda(),
        expected_offset=dp.expected_offset.cuda(),
    )


CUDA_CASES = [_to_cuda(c) for c in manual_cases]


# ==== Helpers ====

def _make_jagged(dp: SuffixDatapoint) -> ITokenMerger.JaggedTokens:
    return ITokenMerger.JaggedTokens(
        tokens=dp.input_tokens, weight=dp.input_weight, offset=dp.input_offset,
    )


def _build_random_jagged(
    M: int, C: int, min_len: int, max_len: int, device: str,
) -> ITokenMerger.JaggedTokens:
    """Build a JaggedTokens with M segments of random lengths in [min_len, max_len]."""
    lengths = torch.randint(min_len, max_len + 1, (M,))
    P = int(lengths.sum().item())
    offset = torch.zeros(M + 1, dtype=torch.long, device=device)
    offset[1:] = lengths.cumsum(0).to(device)
    tokens = torch.randn(1, P, C, device=device)
    weight = torch.randint(1, 5, (P,), device=device).float()
    return ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


# ==== Hand-crafted add_suffix tests ====

@pytest.mark.parametrize("case", CUDA_CASES)
def test_add_suffix_manual(case: SuffixDatapoint):
    jagged = _make_jagged(case)
    result = jagged.add_suffix(case.suffix)

    torch.testing.assert_close(result.tokens, case.expected_tokens)
    torch.testing.assert_close(result.weight, case.expected_weight)
    torch.testing.assert_close(result.offset, case.expected_offset)


# ==== Roundtrip tests (manual) ====

@pytest.mark.parametrize("case", CUDA_CASES)
def test_roundtrip_manual(case: SuffixDatapoint):
    jagged = _make_jagged(case)
    T = case.suffix.shape[1]

    with_suffix = jagged.add_suffix(case.suffix)
    recovered, recovered_suffix = with_suffix.split_suffix(T)

    torch.testing.assert_close(recovered.tokens, jagged.tokens)
    torch.testing.assert_close(recovered.weight, jagged.weight)
    torch.testing.assert_close(recovered.offset, jagged.offset)
    torch.testing.assert_close(recovered_suffix, case.suffix)


# ==== Randomised roundtrip tests ====

_RANDOM_CONFIGS = [
    # (M, T, C, min_len, max_len)
    (1,  1,  8,  1,   1),
    (1,  4, 16,  5,  20),
    (4,  1, 32,  1,  10),
    (4,  3, 64,  2,  15),
    (8,  2, 16,  3,  30),
    (16, 5, 128, 1,  50),
    (32, 1,  8, 10,  10),
]


@pytest.mark.parametrize("M,T,C,lo,hi", _RANDOM_CONFIGS)
def test_roundtrip_random(M: int, T: int, C: int, lo: int, hi: int):
    jagged = _build_random_jagged(M, C, lo, hi, DEVICE)
    suffix = torch.randn(M, T, C, device=DEVICE)

    with_suffix = jagged.add_suffix(suffix)

    assert with_suffix.tokens.shape[1] == jagged.tokens.shape[1] + M * T
    assert with_suffix.offset.shape[0] == M + 1

    recovered, recovered_suffix = with_suffix.split_suffix(T)

    torch.testing.assert_close(recovered.tokens, jagged.tokens)
    torch.testing.assert_close(recovered.weight, jagged.weight)
    torch.testing.assert_close(recovered.offset, jagged.offset)
    torch.testing.assert_close(recovered_suffix, suffix)


# ==== Offset / shape invariant tests ====

@pytest.mark.parametrize("M,T,C,lo,hi", _RANDOM_CONFIGS)
def test_offset_invariants(M: int, T: int, C: int, lo: int, hi: int):
    jagged = _build_random_jagged(M, C, lo, hi, DEVICE)
    suffix = torch.randn(M, T, C, device=DEVICE)
    result = jagged.add_suffix(suffix)

    old_lens = jagged.offset[1:] - jagged.offset[:-1]
    new_lens = result.offset[1:] - result.offset[:-1]
    torch.testing.assert_close(new_lens, old_lens + T)

    assert result.offset[0] == 0
    assert result.offset[-1] == jagged.tokens.shape[1] + M * T


# ==== Edge cases ====

def test_single_element_segments():
    """Each segment has exactly 1 token."""
    M, T, C = 4, 2, 3
    tokens = torch.randn(1, M, C, device=DEVICE)
    weight = torch.ones(M, device=DEVICE)
    offset = torch.arange(M + 1, dtype=torch.long, device=DEVICE)
    jagged = ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)
    suffix = torch.randn(M, T, C, device=DEVICE)

    result = jagged.add_suffix(suffix)
    assert result.tokens.shape[1] == M + M * T

    recovered, recovered_suffix = result.split_suffix(T)
    torch.testing.assert_close(recovered.tokens, jagged.tokens)
    torch.testing.assert_close(recovered_suffix, suffix)


def test_uniform_segment_lengths():
    """All segments have the same length."""
    M, L, T, C = 5, 8, 3, 16
    P = M * L
    tokens = torch.randn(1, P, C, device=DEVICE)
    weight = torch.ones(P, device=DEVICE)
    offset = torch.arange(M + 1, dtype=torch.long, device=DEVICE) * L
    jagged = ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)
    suffix = torch.randn(M, T, C, device=DEVICE)

    result = jagged.add_suffix(suffix)
    recovered, recovered_suffix = result.split_suffix(T)
    torch.testing.assert_close(recovered.tokens, jagged.tokens)
    torch.testing.assert_close(recovered_suffix, suffix)


def test_single_segment():
    """Only one segment — trivial case."""
    T, C = 4, 8
    tokens = torch.randn(1, 10, C, device=DEVICE)
    weight = torch.ones(10, device=DEVICE)
    offset = torch.tensor([0, 10], dtype=torch.long, device=DEVICE)
    jagged = ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)
    suffix = torch.randn(1, T, C, device=DEVICE)

    result = jagged.add_suffix(suffix)
    assert result.tokens.shape[1] == 14
    assert result.offset.tolist() == [0, 14]

    recovered, recovered_suffix = result.split_suffix(T)
    torch.testing.assert_close(recovered.tokens, jagged.tokens)
    torch.testing.assert_close(recovered_suffix, suffix)


# ==== Validation tests ====

def test_add_suffix_batch_mismatch():
    tokens = torch.randn(1, 6, 4, device=DEVICE)
    weight = torch.ones(6, device=DEVICE)
    offset = torch.tensor([0, 3, 6], dtype=torch.long, device=DEVICE)
    jagged = ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)

    bad_suffix = torch.randn(3, 2, 4, device=DEVICE)
    with pytest.raises(ValueError, match="suffix batch dim"):
        jagged.add_suffix(bad_suffix)


def test_split_suffix_exceeds_segment():
    """Segment 0 has length 1, so count=2 exceeds it even though total allows it."""
    tokens = torch.randn(1, 6, 4, device=DEVICE)
    weight = torch.ones(6, device=DEVICE)
    offset = torch.tensor([0, 1, 6], dtype=torch.long, device=DEVICE)
    jagged = ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)

    with pytest.raises(ValueError, match="exceeds the length"):
        jagged.split_suffix(2)


def test_split_suffix_exceeds_total():
    tokens = torch.randn(1, 4, 4, device=DEVICE)
    weight = torch.ones(4, device=DEVICE)
    offset = torch.tensor([0, 2, 4], dtype=torch.long, device=DEVICE)
    jagged = ITokenMerger.JaggedTokens(tokens=tokens, weight=weight, offset=offset)

    with pytest.raises(ValueError, match="Cannot strip"):
        jagged.split_suffix(100)
