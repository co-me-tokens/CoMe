import torch
from typing import TypeAlias, Literal

from ....cuda_extension import (
    flash_attn_varlen_qkvpacked_func_w_perkey_bias,
    flash_attn_varlen_qkvpacked_func,
)
from ....cuda_extension.flash_attn import infer_flash_attn_varlen_max_seqlen
from ....thirdparty.pi3.models.layers.attention import AttentionRope
from ....interface.token_merger import ITokenMerger
from ...common.qkv import prepare_packed_self_attention_qkv


_FlashAttn_Type = Literal["bf16", "fp16"]


class JaggedAttention_w_Bias(torch.nn.Module):
    """Flash-attention-backed attention with per-key additive bias for jagged sequences."""
    SupportType: TypeAlias = _FlashAttn_Type

    def __init__(self, original_attn: AttentionRope, run_dtype: SupportType):
        super().__init__()
        self.run_dtype: torch.dtype
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

        out: torch.Tensor = flash_attn_varlen_qkvpacked_func_w_perkey_bias(  # type: ignore[assignment]
            qkv, cu_seqlens, max_seqlen,
            perkey_bias=perkey_bias,
            dropout_p=dropout_p, softmax_scale=attn.scale,
        ) # (P, H, D)

        out = out.reshape(P, C)
        out = attn.proj(out)
        out = attn.proj_drop(out)

        return ITokenMerger.JaggedTokens(tokens=out.unsqueeze(0), weight=x.weight, offset=x.offset)


class JaggedAttention(torch.nn.Module):
    SupportType: TypeAlias = _FlashAttn_Type
    
    def __init__(self, original_attn: AttentionRope, run_dtype: SupportType):
        super().__init__()
        self.run_dtype: torch.dtype
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

        out: torch.Tensor = flash_attn_varlen_qkvpacked_func(   # type: ignore
            qkv, cu_seqlens, max_seqlen,
            dropout_p=dropout_p, softmax_scale=attn.scale,
        ) # (P, H, D)

        out = out.reshape(P, C)
        out = attn.proj(out)
        out = attn.proj_drop(out)

        return ITokenMerger.JaggedTokens(tokens=out.unsqueeze(0), weight=x.weight, offset=x.offset)
