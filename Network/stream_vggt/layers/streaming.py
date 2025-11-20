from typing import Callable
import torch
from torch.nn.modules import Module

from .mlp import Mlp
from .block     import Block
from .attention import Attention
from .auto_scaling_tensor import AutoScalingTensor
from ..models.interface import StreamLike

from Accelerate.common import sdpa_dispatch


class StreamAttention(Attention, StreamLike):
    def __init__(
        self, dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0,
        proj_drop: float = 0,
        norm_layer: Module = torch.nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop, norm_layer, qk_norm, fused_attn, rope)
        
        self._cache_init_shape = 8192 * 8
        self._kv_cache         = None
    
    def _init_stream(self, k: torch.Tensor, v: torch.Tensor):
        B, H, _, C = k.shape
        self._kv_cache = (
            AutoScalingTensor(torch.empty(B, H, self._cache_init_shape, C, device=k.device, dtype=k.dtype), grow_dim=2),
            AutoScalingTensor(torch.empty(B, H, self._cache_init_shape, C, device=k.device, dtype=k.dtype), grow_dim=2),
            AutoScalingTensor(torch.empty(B, 1, 1, self._cache_init_shape, device=k.device, dtype=k.dtype), grow_dim=-1)
        )
    
    def succ(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        B, new_N, C = x.shape
        
        with torch.cuda.nvtx.range("QKV Projection"):
            qkv = self.qkv(x).reshape(B, new_N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        new_q, new_k, new_v = qkv.unbind(0)
        new_q, new_k = self.q_norm(new_q), self.k_norm(new_k)
        
        with torch.cuda.nvtx.range("RoPE"):
            if self.rope is not None:
                new_q = self.rope(new_q, pos)
                new_k = self.rope(new_k, pos)
        
        new_q, new_k, new_v = new_q.bfloat16(), new_k.bfloat16(), new_v.bfloat16()
        
        with torch.cuda.nvtx.range("KV Cache"):
            # KV Cache Management
            if self._kv_cache is None: self._init_stream(new_k, new_v)
            
            assert self._kv_cache is not None
            self._kv_cache[0].push(new_k)
            self._kv_cache[1].push(new_v)
        
            q, k, v = new_q, self._kv_cache[0].data, self._kv_cache[1].data
                
        with torch.cuda.nvtx.range("SDPA"):
            if self.fused_attn:
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
                x = sdpa_dispatch(
                    q, k, v, attn_bias=None, backend="torch-flex"
                )
                
                x = x.to(qkv)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
        
        with torch.cuda.nvtx.range("Projection"):
            x = x.transpose(1, 2).reshape(B, new_N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        return x
    
    def reset(self) -> None:
        self._kv_cache = None


class StreamBlock(Block, StreamLike):
    def __init__(
        self, dim: int,
        num_heads: int,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0,
        attn_drop: float = 0,
        init_values=None,
        drop_path: float = 0,
        act_layer: Callable[..., Module] = torch.nn.GELU,
        norm_layer: Callable[..., Module] = torch.nn.LayerNorm,
        attn_class = StreamAttention,
        ffn_layer = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None
    ) -> None:
        self.attn: StreamAttention
        assert isinstance(attn_class, type) and issubclass(attn_class, StreamAttention)
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, attn_class, ffn_layer, qk_norm, fused_attn, rope)
    
    def succ(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor, pos=None) -> torch.Tensor:
            with torch.cuda.nvtx.range("Attn"):
                return self.ls1(self.attn.succ(self.norm1(x), pos=pos))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            with torch.cuda.nvtx.range("MLP"):
                return self.ls2(self.mlp(self.norm2(x)))

        with torch.cuda.nvtx.range("Block"):
            if self.training and self.sample_drop_ratio > 0.1:
                # the overhead is compensated only for a drop path rate larger than 0.1
                x = drop_add_residual_stochastic_depth(
                    x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
                )
                x = drop_add_residual_stochastic_depth(
                    x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
                )
            elif self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path1(attn_residual_func(x, pos=pos))
                x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
            else:
                x = x + attn_residual_func(x, pos=pos)
                x = x + ffn_residual_func(x)
        return x

    def reset(self) -> None:
        self.attn.reset()
