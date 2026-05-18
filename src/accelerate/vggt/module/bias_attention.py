"""
Attention with per-key bias correction for variable-length (jagged) sequences.

Contains:
  - naive_jagged_attention_with_bias: FlexAttention reference (slow, for testing)
  - JaggedAttention_w_Bias: CUDA flash-attention-backed module (fast, for production)
"""
import os
from typing import Literal, TypeAlias

import torch
import jaxtyping as jt
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from ....cuda_extension import flash_attn_varlen_qkvpacked_func_w_perkey_bias
from ....cuda_extension.flash_attn import infer_flash_attn_varlen_max_seqlen
from ....thirdparty.vggt.layers.attention import Attention
from ....interface.token_merger import ITokenMerger
from ...common.qkv import prepare_packed_self_attention_qkv

from .plain_attention import JaggedAttention


os.environ['TRITON_DYNAMIC_SHAPES'] = '1'

FLEX_BLOCK_SIZE = 128

fn_flex_attention = torch.compile(
    flex_attention,
    mode='reduce-overhead',
    dynamic=True
)


def _build_document_id(offset: jt.Int64[torch.Tensor, "M"], P_padded: int) -> jt.Int64[torch.Tensor, "Pp"]:
    """Map each packed-sequence position to its segment index; padding gets -1."""
    P = int(offset[-1].item())
    positions = torch.arange(P_padded, device=offset.device)
    doc_id = torch.searchsorted(offset, positions, right=True) - 1
    doc_id[P:] = -1
    return doc_id


def naive_jagged_attention_with_bias(
    tokens: ITokenMerger.JaggedTokens,
    qkv: jt.Float[torch.Tensor, "P 3 H D"]
) -> jt.Float[torch.Tensor, "P H D"]:
    P, _, H, D = qkv.shape
    device = qkv.device

    P_padded = P + (-P % FLEX_BLOCK_SIZE)

    q, k, v = qkv.unbind(dim=1)

    if P_padded > P:
        pad = torch.zeros(P_padded - P, H, D, dtype=qkv.dtype, device=device)
        q = torch.cat([q, pad], dim=0)
        k = torch.cat([k, pad], dim=0)
        v = torch.cat([v, pad], dim=0)

    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)
    v = v.unsqueeze(0).transpose(1, 2)

    document_id = _build_document_id(tokens.offset, P_padded)

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return document_id[q_idx] == document_id[kv_idx]

    block_mask = create_block_mask(mask_mod, B=1, H=None, Q_LEN=P_padded, KV_LEN=P_padded, device=str(device))  #type: ignore

    log_weight = torch.zeros(P_padded, dtype=qkv.dtype, device=device)
    log_weight[:P] = tokens.weight.to(dtype=qkv.dtype).log()

    def score_mod(score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return score + log_weight[kv_idx]

    out: torch.Tensor = fn_flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)  # type: ignore[assignment]

    return out.transpose(1, 2).squeeze(0)[:P]


class JaggedAttention_w_Bias(torch.nn.Module):
    """Flash-attention-backed attention with per-key additive bias for jagged sequences."""
    SupportType: TypeAlias = JaggedAttention.SupportType
    AttnBackend: TypeAlias = Literal["flash_attn", "flex_attn"]

    def __init__(self, original_attn: Attention, run_dtype: SupportType, backend: AttnBackend = "flash_attn"):
        super().__init__()
        self.run_dtype: torch.dtype
        self.backend  : JaggedAttention_w_Bias.AttnBackend = backend

        match run_dtype:
            case "bf16": self.run_dtype = torch.bfloat16
            case "fp16": self.run_dtype = torch.float16
            case _: raise ValueError(f"FlashAttention can only run on fp16/bf16, get {_}")

        self.attn = original_attn.to(dtype=self.run_dtype)

    def forward(self, x: ITokenMerger.JaggedTokens, pos: torch.Tensor | None = None) -> ITokenMerger.JaggedTokens:
        attn = self.attn
        tokens = x.tokens.squeeze(0)                          # (P, C)
        P, C = tokens.shape

        qkv = attn.qkv(tokens)                                # (P, 3C)
        qkv = qkv.reshape(P, 3, attn.num_heads, attn.head_dim)
        qkv = prepare_packed_self_attention_qkv(
            qkv=qkv,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            rope=attn.rope,
            pos=pos,
            run_dtype=self.run_dtype,
        )

        cu_seqlens = x.offset.to(torch.int32)
        max_seqlen = infer_flash_attn_varlen_max_seqlen(cu_seqlens)
        dropout_p  = attn.attn_drop.p if self.training else 0.0

        perkey_bias = x.weight.float().log().to(dtype=self.run_dtype)

        match self.backend:
            case "flash_attn":
                out: torch.Tensor = flash_attn_varlen_qkvpacked_func_w_perkey_bias(  # type: ignore[assignment]
                    qkv, cu_seqlens, max_seqlen,
                    perkey_bias=perkey_bias,
                    dropout_p=dropout_p, softmax_scale=attn.scale,
                ) # (P, H, D)
            case "flex_attn":
                out: torch.Tensor = naive_jagged_attention_with_bias(x, qkv)


        out = out.reshape(P, C)
        out = attn.proj(out)
        out = attn.proj_drop(out)

        return ITokenMerger.JaggedTokens(tokens=out.unsqueeze(0), weight=x.weight, offset=x.offset)


if __name__ == "__main__":
    device = torch.device("cuda")
    tokens = ITokenMerger.JaggedTokens(
        tokens=torch.randn(1, 128, 256, device=device),
        weight=torch.rand(128, device=device).clamp(min=1e-3),
        offset=torch.tensor([0, 56, 105, 128], dtype=torch.long, device=device),
    )
    qkv = torch.randn(128, 3, 1, 64, device=device)
    result = naive_jagged_attention_with_bias(tokens, qkv)
    print(f"Output shape: {result.shape}")
