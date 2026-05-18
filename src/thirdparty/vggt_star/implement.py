import torch
import typing as tp

from ..vggt.models.vggt      import VGGT
from ..vggt.layers.block     import Block
from ..vggt.layers.attention import Attention


class EnforceFlashAttention(torch.nn.Module):
    def __init__(self, attention: Attention, run_dtype: torch.dtype):
        super().__init__()
        self.run_dtype = run_dtype
        self.attn      = attention

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        attn      = self.attn
        out_dtype = x.dtype

        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)

        if attn.rope is not None:
            q = attn.rope(q, pos)
            k = attn.rope(k, pos)

        # Flash attention requires fp16 / bf16; cast q, k, v only, not the full x
        q = q.to(dtype=self.run_dtype)
        k = k.to(dtype=self.run_dtype)
        v = v.to(dtype=self.run_dtype)

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=attn.attn_drop.p if attn.training else 0.0,
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn.proj(x.to(dtype=out_dtype))
        x = attn.proj_drop(x)
        return x


def enforce_flashattn_vggt(model: VGGT) -> VGGT:
    dtype = torch.bfloat16
    
    for blk in model.aggregator.patch_embed.blocks:
        blk = tp.cast(Block, blk)
        blk.attn = EnforceFlashAttention(blk.attn, dtype)
    
    for blk in model.aggregator.frame_blocks:
        blk = tp.cast(Block, blk)
        blk.attn = EnforceFlashAttention(blk.attn, dtype)
    
    for blk in model.aggregator.global_blocks:
        blk = tp.cast(Block, blk)
        blk.attn = EnforceFlashAttention(blk.attn, dtype)
    
    return model
