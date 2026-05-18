"""Correctness tests for attention with per-key bias.

Compares:
  1. FlexAttention-based naive_jagged_attention_with_bias against pure-PyTorch reference
  2. CUDA flash_attn_varlen_qkvpacked_func_w_perkey_bias against the same reference
"""

import pytest
import torch

from src.accelerate.vggt.module.bias_attention import (
    naive_jagged_attention_with_bias,
    _build_document_id,
)
from src.cuda_extension import flash_attn_varlen_qkvpacked_func_w_perkey_bias
from src.interface.token_merger import ITokenMerger


DEVICE = torch.device("cuda")
ATOL   = 1e-2
RTOL   = 1e-2


# ==== Helpers ====

def _reference_attention_with_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_weight: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pure-PyTorch scaled dot-product attention with per-key additive bias (single head)."""
    attn = (q @ k.T) * scale + log_weight[None, :]
    attn = torch.softmax(attn, dim=-1)
    return attn @ v


def _make_jagged_tokens(
    seq_lens: list[int],
    device: torch.device = DEVICE,
) -> ITokenMerger.JaggedTokens:
    P = sum(seq_lens)
    offsets = torch.zeros(len(seq_lens) + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.tensor(seq_lens, device=device).cumsum(0)
    return ITokenMerger.JaggedTokens(
        tokens=torch.randn(1, P, 64, device=device),
        weight=torch.rand(P, device=device).clamp(min=1e-3),
        offset=offsets,
    )


# ==== _build_document_id tests ====

def test_build_document_id_basic():
    offset = torch.tensor([0, 3, 5, 8], dtype=torch.long, device=DEVICE)
    doc_id = _build_document_id(offset, 8)
    expected = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long, device=DEVICE)
    torch.testing.assert_close(doc_id, expected)


def test_build_document_id_with_padding():
    offset = torch.tensor([0, 3, 5], dtype=torch.long, device=DEVICE)
    doc_id = _build_document_id(offset, 8)
    expected = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1], dtype=torch.long, device=DEVICE)
    torch.testing.assert_close(doc_id, expected)


def test_build_document_id_single_segment():
    offset = torch.tensor([0, 4], dtype=torch.long, device=DEVICE)
    doc_id = _build_document_id(offset, 4)
    expected = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=DEVICE)
    torch.testing.assert_close(doc_id, expected)


# ==== FlexAttention correctness tests ====

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def _assert_segments_match_reference(
    seq_lens: list[int],
    H: int,
    D: int,
) -> None:
    """Run jagged attention and compare each segment against the standalone reference."""
    P = sum(seq_lens)
    tokens = _make_jagged_tokens(seq_lens)
    qkv = torch.randn(P, 3, H, D, device=DEVICE)

    result = naive_jagged_attention_with_bias(tokens, qkv)

    q_all, k_all, v_all = qkv.unbind(dim=1)
    log_w = tokens.weight.to(dtype=qkv.dtype).log()
    scale = 1.0 / (D ** 0.5)
    offset = tokens.offset

    for i in range(len(seq_lens)):
        s = int(offset[i].item())
        e = int(offset[i + 1].item())
        for h in range(H):
            ref = _reference_attention_with_bias(
                q_all[s:e, h, :], k_all[s:e, h, :], v_all[s:e, h, :],
                log_w[s:e], scale,
            )
            torch.testing.assert_close(result[s:e, h, :], ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("seq_lens", [
    [32, 48, 24, 24],
    [56, 49, 23],
    [128],
])
def test_matches_reference_block_aligned(seq_lens: list[int]):
    """Segments whose total is block-aligned (128) must match standalone reference."""
    _assert_segments_match_reference(seq_lens, H=2, D=32)


def test_matches_reference_needs_padding():
    """When total P is not block-aligned, padding must be transparent to the result."""
    _assert_segments_match_reference([60, 40], H=2, D=32)


def test_matches_reference_single_segment_padded():
    """Single short segment that requires padding."""
    _assert_segments_match_reference([50], H=1, D=64)


def test_output_shape():
    P, H, D = 128, 4, 32
    tokens = _make_jagged_tokens([64, 64])
    qkv = torch.randn(P, 3, H, D, device=DEVICE)
    result = naive_jagged_attention_with_bias(tokens, qkv)
    assert result.shape == (P, H, D)


def test_output_shape_with_padding():
    P, H, D = 100, 2, 32
    tokens = _make_jagged_tokens([60, 40])
    qkv = torch.randn(P, 3, H, D, device=DEVICE)
    result = naive_jagged_attention_with_bias(tokens, qkv)
    assert result.shape == (P, H, D)


def test_uniform_weight_matches_sdpa():
    """When all weights are 1 (log_weight=0), result must match standard SDPA."""
    seq_lens = [64, 64]
    H, D = 2, 32
    P = sum(seq_lens)

    offsets = torch.zeros(3, dtype=torch.long, device=DEVICE)
    offsets[1:] = torch.tensor(seq_lens, device=DEVICE).cumsum(0)
    tokens = ITokenMerger.JaggedTokens(
        tokens=torch.randn(1, P, 64, device=DEVICE),
        weight=torch.ones(P, device=DEVICE),
        offset=offsets,
    )
    qkv = torch.randn(P, 3, H, D, device=DEVICE)

    result = naive_jagged_attention_with_bias(tokens, qkv)

    q_all, k_all, v_all = qkv.unbind(dim=1)
    scale = 1.0 / (D ** 0.5)

    for i in range(len(seq_lens)):
        s, e = int(offsets[i].item()), int(offsets[i + 1].item())
        for h in range(H):
            q_seg = q_all[s:e, h, :].unsqueeze(0)
            k_seg = k_all[s:e, h, :].unsqueeze(0)
            v_seg = v_all[s:e, h, :].unsqueeze(0)
            ref = torch.nn.functional.scaled_dot_product_attention(q_seg, k_seg, v_seg, scale=scale)
            torch.testing.assert_close(result[s:e, h, :].unsqueeze(0), ref, atol=ATOL, rtol=RTOL)


def test_bias_has_effect():
    """Different weights must produce different outputs for the same Q/K/V."""
    P, H, D = 128, 1, 64
    offsets = torch.tensor([0, 128], dtype=torch.long, device=DEVICE)
    qkv = torch.randn(P, 3, H, D, device=DEVICE)

    tokens_uniform = ITokenMerger.JaggedTokens(
        tokens=torch.randn(1, P, 64, device=DEVICE),
        weight=torch.ones(P, device=DEVICE),
        offset=offsets,
    )

    weight_biased = torch.ones(P, device=DEVICE)
    weight_biased[:64] = 2.0
    tokens_biased = ITokenMerger.JaggedTokens(
        tokens=torch.randn(1, P, 64, device=DEVICE),
        weight=weight_biased,
        offset=offsets,
    )

    out_uniform = naive_jagged_attention_with_bias(tokens_uniform, qkv).clone()
    out_biased  = naive_jagged_attention_with_bias(tokens_biased, qkv)

    assert not torch.allclose(out_uniform, out_biased, atol=1e-4)


# ==== CUDA flash attention with per-key bias tests ====

def _assert_cuda_bias_matches_reference(
    seq_lens: list[int],
    H: int,
    D: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Run CUDA flash attention w/ bias and compare each segment against the pure-PyTorch reference."""
    P = sum(seq_lens)
    offsets = torch.zeros(len(seq_lens) + 1, dtype=torch.long, device=DEVICE)
    offsets[1:] = torch.tensor(seq_lens, device=DEVICE).cumsum(0)
    weight = torch.rand(P, device=DEVICE).clamp(min=1e-3)

    qkv = torch.randn(P, 3, H, D, device=DEVICE, dtype=dtype)
    cu_seqlens = offsets.to(torch.int32)
    max_seqlen = max(seq_lens)
    perkey_bias = weight.float().log().to(dtype)
    scale = 1.0 / (D ** 0.5)

    result = flash_attn_varlen_qkvpacked_func_w_perkey_bias(
        qkv, cu_seqlens, max_seqlen,
        perkey_bias=perkey_bias,
        softmax_scale=scale,
    )
    assert isinstance(result, torch.Tensor)

    q_all, k_all, v_all = qkv.float().unbind(dim=1)
    log_w = perkey_bias.float()

    for i in range(len(seq_lens)):
        s = int(offsets[i].item())
        e = int(offsets[i + 1].item())
        for h in range(H):
            ref = _reference_attention_with_bias(
                q_all[s:e, h, :], k_all[s:e, h, :], v_all[s:e, h, :],
                log_w[s:e], scale,
            )
            torch.testing.assert_close(
                result[s:e, h, :].float(), ref, atol=ATOL, rtol=RTOL,
            )


@pytest.mark.parametrize("seq_lens", [
    [32, 48, 24, 24],
    [56, 49, 23],
    [128],
])
def test_cuda_bias_matches_reference_block_aligned(seq_lens: list[int]):
    _assert_cuda_bias_matches_reference(seq_lens, H=2, D=32)


def test_cuda_bias_matches_reference_needs_padding():
    _assert_cuda_bias_matches_reference([60, 40], H=2, D=32)


def test_cuda_bias_matches_reference_single_segment():
    _assert_cuda_bias_matches_reference([50], H=1, D=64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_bias_dtypes(dtype: torch.dtype):
    _assert_cuda_bias_matches_reference([64, 64], H=4, D=64, dtype=dtype)


def test_cuda_bias_output_shape():
    P, H, D = 128, 4, 32
    qkv = torch.randn(P, 3, H, D, device=DEVICE, dtype=torch.float16)
    cu = torch.tensor([0, 64, 128], device=DEVICE, dtype=torch.int32)
    bias = torch.zeros(P, device=DEVICE, dtype=torch.float16)
    result = flash_attn_varlen_qkvpacked_func_w_perkey_bias(qkv, cu, 64, perkey_bias=bias)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (P, H, D)


def test_cuda_uniform_weight_matches_no_bias():
    """When perkey_bias is all zeros, CUDA bias result must match plain flash attention."""
    from src.cuda_extension import flash_attn_varlen_qkvpacked_func

    seq_lens = [64, 64]
    H, D = 4, 64
    P = sum(seq_lens)
    dtype = torch.float16

    qkv = torch.randn(P, 3, H, D, device=DEVICE, dtype=dtype)
    offsets = torch.zeros(3, dtype=torch.long, device=DEVICE)
    offsets[1:] = torch.tensor(seq_lens, device=DEVICE).cumsum(0)
    cu = offsets.to(torch.int32)
    max_seqlen = max(seq_lens)
    scale = 1.0 / (D ** 0.5)

    out_plain = flash_attn_varlen_qkvpacked_func(qkv, cu, max_seqlen, softmax_scale=scale)
    bias_zeros = torch.zeros(P, device=DEVICE, dtype=dtype)
    out_biased = flash_attn_varlen_qkvpacked_func_w_perkey_bias(
        qkv, cu, max_seqlen, perkey_bias=bias_zeros, softmax_scale=scale,
    )
    torch.testing.assert_close(out_plain, out_biased, atol=1e-4, rtol=1e-4)


def test_cuda_bias_has_effect():
    """Non-zero bias must produce different output from zero bias."""
    P, H, D = 128, 2, 64
    dtype = torch.float16
    qkv = torch.randn(P, 3, H, D, device=DEVICE, dtype=dtype)
    cu = torch.tensor([0, 128], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / (D ** 0.5)

    bias_zero = torch.zeros(P, device=DEVICE, dtype=dtype)
    bias_nonzero = torch.randn(P, device=DEVICE, dtype=dtype)

    out_zero = flash_attn_varlen_qkvpacked_func_w_perkey_bias(
        qkv, cu, 128, perkey_bias=bias_zero, softmax_scale=scale,
    )
    out_nonzero = flash_attn_varlen_qkvpacked_func_w_perkey_bias(
        qkv, cu, 128, perkey_bias=bias_nonzero, softmax_scale=scale,
    )
    assert isinstance(out_zero, torch.Tensor)
    assert isinstance(out_nonzero, torch.Tensor)
    assert not torch.allclose(out_zero, out_nonzero, atol=1e-4)


@pytest.mark.parametrize("seq_lens", [
    [128, 256],
    [100, 150, 50],
    [64, 64, 64, 64],
])
def test_cuda_bias_larger_sequences(seq_lens: list[int]):
    _assert_cuda_bias_matches_reference(seq_lens, H=4, D=64)
