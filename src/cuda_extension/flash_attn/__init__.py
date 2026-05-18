"""Forward-only reproduction of flash_attn_varlen_qkvpacked_func."""

from typing import Optional, Tuple

import jaxtyping as Jt
import torch
from . import flash_attn_fwd_cuda as _C


ONNX_DOMAIN = "comesdk"
FLASH_ATTN_VARLEN_QKVPACKED_OP = "FlashAttnVarlenQKVPacked"
FLASH_ATTN_VARLEN_QKVPACKED_W_BIAS_OP = "FlashAttnVarlenQKVPackedWBias"


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is not None and x.stride(-1) != 1:
        return x.contiguous()
    return x


def infer_flash_attn_varlen_max_seqlen(
    cu_seqlens: Jt.Int[torch.Tensor, "batch_plus_1"],
) -> int:
    """Return the eager max sequence length while staying ONNX-export safe.

    The legacy ONNX exporter traces Python integers. Returning a sentinel during
    export avoids specializing the graph to the example jagged layout; the
    custom ONNX node instead consumes ``cu_seqlens`` directly and lets later
    runtimes/plugins interpret the packed layout.
    """
    if torch.onnx.is_in_onnx_export():
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


def _infer_flash_attn_output_type(qkv: torch.Value) -> torch.Type | None:
    qkv_type = qkv.type()
    if not qkv_type.isSubtypeOf(torch._C.TensorType.get()):
        return None

    output_type = qkv_type
    qkv_sizes = qkv_type.sizes()
    if qkv_sizes is not None and len(qkv_sizes) == 4:
        output_type = output_type.with_sizes([qkv_sizes[0], qkv_sizes[2], qkv_sizes[3]])
    return output_type


def _set_output_type_like_qkv(node: torch.Value, qkv: torch.Value) -> torch.Value:
    output_type = _infer_flash_attn_output_type(qkv)
    if output_type is not None:
        node.setType(output_type)
    return node


def _normalize_window_size(window_size: tuple[int, int]) -> tuple[int, int]:
    if len(window_size) != 2:
        raise ValueError(f"window_size must contain 2 values, got {window_size}")
    return int(window_size[0]), int(window_size[1])


def _create_flash_attn_symbolic_node(
    g: torch.Graph,
    op_name: str,
    qkv: torch.Value,
    cu_seqlens: torch.Value,
    perkey_bias: torch.Value | None,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Value | None,
    return_softmax: bool,
) -> torch.Value:
    if return_softmax:
        raise RuntimeError(f"{op_name} ONNX export does not support return_attn_probs=True yet.")
    if alibi_slopes is not None:
        raise RuntimeError(f"{op_name} ONNX export does not support alibi_slopes yet.")

    window_size_left, window_size_right = _normalize_window_size(window_size)
    inputs: list[torch.Value] = [qkv, cu_seqlens]
    if perkey_bias is not None:
        inputs.append(perkey_bias)

    node = g.op(
        f"{ONNX_DOMAIN}::{op_name}",
        *inputs,
        dropout_p_f=float(dropout_p),
        softmax_scale_f=0.0 if softmax_scale is None else float(softmax_scale),
        causal_i=int(bool(causal)),
        window_size_left_i=window_size_left,
        window_size_right_i=window_size_right,
        softcap_f=float(softcap),
    )
    return _set_output_type_like_qkv(node, qkv)


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = _C.varlen_fwd(
        q,
        k,
        v,
        None,          # out_
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,          # gen_
    )
    return out, softmax_lse, S_dmask, rng_state


class _FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        ctx.mark_non_differentiable(softmax_lse, S_dmask, rng_state)
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError(
            "flash_attn_varlen_qkvpacked_func backward is not supported "
            "in this forward-only build. Use the full flash_attn package "
            "if you need gradient computation."
        )

    @staticmethod
    def symbolic(
        g: torch.Graph,
        qkv: torch.Value,
        cu_seqlens: torch.Value,
        max_seqlen,
        dropout_p: float,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
        softcap: float,
        alibi_slopes,
        deterministic: bool,
        return_softmax: bool,
        is_grad_enabled: bool,
    ) -> torch.Value:
        del max_seqlen, deterministic, is_grad_enabled
        return _create_flash_attn_symbolic_node(
            g,
            FLASH_ATTN_VARLEN_QKVPACKED_OP,
            qkv,
            cu_seqlens,
            perkey_bias=None,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )


def flash_attn_varlen_qkvpacked_func(
    qkv: Jt.Float[torch.Tensor, "total 3 nheads headdim"],
    cu_seqlens: Jt.Int32[torch.Tensor, "batch_plus_1"],
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Jt.Float32[torch.Tensor, "*batch nheads"] | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    NOTE: This is a forward-only build. Calling backward will raise RuntimeError.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. (ignored — no backward pass in this build).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
            testing only. The returned probabilities are not guaranteed to be correct
            (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
    """
    return _FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def _flash_attn_varlen_forward_w_perkey_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    perkey_bias: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = _C.varlen_fwd_w_perkey_bias(
        q,
        k,
        v,
        None,          # out_
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        perkey_bias,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,          # gen_
    )
    return out, softmax_lse, S_dmask, rng_state


class _FlashAttnVarlenQKVPackedFuncWBias(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        perkey_bias,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward_w_perkey_bias(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            perkey_bias,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        ctx.mark_non_differentiable(softmax_lse, S_dmask, rng_state)
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError(
            "flash_attn_varlen_qkvpacked_func_w_perkey_bias backward is not supported "
            "in this forward-only build."
        )

    @staticmethod
    def symbolic(
        g: torch.Graph,
        qkv: torch.Value,
        cu_seqlens: torch.Value,
        max_seqlen,
        perkey_bias: torch.Value,
        dropout_p: float,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
        softcap: float,
        alibi_slopes,
        deterministic: bool,
        return_softmax: bool,
        is_grad_enabled: bool,
    ) -> torch.Value:
        del max_seqlen, deterministic, is_grad_enabled
        return _create_flash_attn_symbolic_node(
            g,
            FLASH_ATTN_VARLEN_QKVPACKED_W_BIAS_OP,
            qkv,
            cu_seqlens,
            perkey_bias=perkey_bias,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )


def flash_attn_varlen_qkvpacked_func_w_perkey_bias(
    qkv: Jt.Float[torch.Tensor, "total 3 nheads headdim"],
    cu_seqlens: Jt.Int32[torch.Tensor, "batch_plus_1"],
    max_seqlen: int,
    perkey_bias: Jt.Float[torch.Tensor, "total_k"],
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Jt.Float32[torch.Tensor, "*batch nheads"] | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as flash_attn_varlen_qkvpacked_func but with an additive per-key
    bias applied to the scaled attention scores before softmax.

    The kernel computes: softmax(Q @ K.T * scale + perkey_bias[kv_idx]) @ V

    Arguments:
        perkey_bias: (total,) tensor of additive biases, same dtype as qkv
            (fp16 or bf16). Indexed identically to the packed K/V tokens
            (via cu_seqlens).
    """
    return _FlashAttnVarlenQKVPackedFuncWBias.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        perkey_bias,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


__all__ = [
    "FLASH_ATTN_VARLEN_QKVPACKED_OP",
    "FLASH_ATTN_VARLEN_QKVPACKED_W_BIAS_OP",
    "ONNX_DOMAIN",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_qkvpacked_func_w_perkey_bias",
    "infer_flash_attn_varlen_max_seqlen",
]
