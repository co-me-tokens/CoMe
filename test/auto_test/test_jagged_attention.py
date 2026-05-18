"""JaggedAttention correctness tests.

For each configuration, constructs B samples of varying lengths, runs
JaggedAttention on the concatenated jagged input, then runs the original
Attention on each sample individually (batch size 1).  The per-sample
results must match within floating-point tolerance.
"""

import pytest
import torch

from src.accelerate.vggt.module import JaggedAttention
from src.interface.token_merger import ITokenMerger
from src.thirdparty.vggt.layers.attention import Attention
from src.thirdparty.vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D

DEVICE = torch.device("cuda")
DTYPE  = torch.float16
ATOL   = 1e-3
RTOL   = 1e-3


# ==== Helpers ====

def _build_jagged(
    samples: list[torch.Tensor],
) -> tuple[ITokenMerger.JaggedTokens, list[int]]:
    """Pack a list of (1, N_i, C) tensors into a JaggedTokens datapack."""
    seq_lens = [s.shape[1] for s in samples]
    flat = torch.cat([s.squeeze(0) for s in samples], dim=0)
    offsets = torch.zeros(len(seq_lens) + 1, dtype=torch.int64, device=flat.device)
    offsets[1:] = torch.tensor(seq_lens, device=flat.device).cumsum(0)
    weight = torch.ones(flat.shape[0], dtype=torch.float, device=flat.device)
    return ITokenMerger.JaggedTokens(tokens=flat.unsqueeze(0), weight=weight, offset=offsets), seq_lens


def _per_sample_forward(
    attn: Attention,
    samples: list[torch.Tensor],
    positions: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Run naive attention on each sample independently (B=1)."""
    results = []
    for i, s in enumerate(samples):
        pos = positions[i] if positions is not None else None
        results.append(attn(s, pos=pos))
    return results


def _compare(
    jagged_out: ITokenMerger.JaggedTokens,
    per_sample: list[torch.Tensor],
    seq_lens: list[int],
) -> None:
    offset = jagged_out.offset
    for i, (ref, n) in enumerate(zip(per_sample, seq_lens)):
        start, end = int(offset[i].item()), int(offset[i + 1].item())
        actual = jagged_out.tokens[0, start:end, :]
        torch.testing.assert_close(actual, ref.squeeze(0), atol=ATOL, rtol=RTOL)


# ==== Test cases ====

DIM       = 64
NUM_HEADS = 4
SEQ_LENS  = [10, 20, 15, 8]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def test_basic_no_rope_no_qknorm():
    attn = Attention(DIM, num_heads=NUM_HEADS, qk_norm=False, rope=None).to(DEVICE, DTYPE).eval()
    jagged_attn = JaggedAttention(attn, run_dtype='fp16').to(DEVICE, DTYPE).eval()

    samples = [torch.randn(1, n, DIM, device=DEVICE, dtype=DTYPE) for n in SEQ_LENS]
    jagged, lens = _build_jagged(samples)

    with torch.no_grad():
        out = jagged_attn(jagged)
        ref = _per_sample_forward(attn, samples)

    _compare(out, ref, lens)


def test_with_qknorm():
    attn = Attention(DIM, num_heads=NUM_HEADS, qk_norm=True, rope=None).to(DEVICE, DTYPE).eval()
    jagged_attn = JaggedAttention(attn, run_dtype='fp16').to(DEVICE, DTYPE).eval()

    samples = [torch.randn(1, n, DIM, device=DEVICE, dtype=DTYPE) for n in SEQ_LENS]
    jagged, lens = _build_jagged(samples)

    with torch.no_grad():
        out = jagged_attn(jagged)
        ref = _per_sample_forward(attn, samples)

    _compare(out, ref, lens)


def test_with_rope():
    rope = RotaryPositionEmbedding2D()
    attn = Attention(DIM, num_heads=NUM_HEADS, qk_norm=False, rope=rope).to(DEVICE, DTYPE).eval()
    jagged_attn = JaggedAttention(attn, run_dtype='fp16').to(DEVICE, DTYPE).eval()

    pos_getter = PositionGetter()
    heights_widths = [(2, 5), (4, 5), (3, 5), (2, 4)]
    samples = []
    positions = []
    for h, w in heights_widths:
        n = h * w
        samples.append(torch.randn(1, n, DIM, device=DEVICE, dtype=DTYPE))
        positions.append(pos_getter(1, h, w, DEVICE))

    jagged, lens = _build_jagged(samples)
    jagged_pos = torch.cat([p.squeeze(0) for p in positions], dim=0).unsqueeze(0)

    with torch.no_grad():
        out = jagged_attn(jagged, pos=jagged_pos)
        ref = _per_sample_forward(attn, samples, positions)

    _compare(out, ref, lens)


def test_single_sample():
    attn = Attention(DIM, num_heads=NUM_HEADS, qk_norm=False, rope=None).to(DEVICE, DTYPE).eval()
    jagged_attn = JaggedAttention(attn, run_dtype='fp16').to(DEVICE, DTYPE).eval()

    samples = [torch.randn(1, 25, DIM, device=DEVICE, dtype=DTYPE)]
    jagged, lens = _build_jagged(samples)

    with torch.no_grad():
        out = jagged_attn(jagged)
        ref = _per_sample_forward(attn, samples)

    _compare(out, ref, lens)


def test_uniform_length():
    attn = Attention(DIM, num_heads=NUM_HEADS, qk_norm=False, rope=None).to(DEVICE, DTYPE).eval()
    jagged_attn = JaggedAttention(attn, run_dtype='fp16').to(DEVICE, DTYPE).eval()

    n = 16
    samples = [torch.randn(1, n, DIM, device=DEVICE, dtype=DTYPE) for _ in range(4)]
    jagged, lens = _build_jagged(samples)

    with torch.no_grad():
        out = jagged_attn(jagged)
        ref = _per_sample_forward(attn, samples)

    _compare(out, ref, lens)
