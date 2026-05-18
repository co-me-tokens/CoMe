"""Utilities for packed QKV self-attention preparation."""

import typing as T
import torch


def prepare_packed_self_attention_qkv(
    qkv: torch.Tensor,
    q_norm: T.Callable[[torch.Tensor], torch.Tensor],
    k_norm: T.Callable[[torch.Tensor], torch.Tensor],
    rope: T.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    pos: torch.Tensor | None,
    run_dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize packed Q/K and optionally apply RoPE.

    During inference we update the packed tensor in-place to avoid rebuilding it
    with ``torch.stack``. When gradients are enabled we must keep the out-of-place
    path because mutating the normalized views breaks autograd version tracking.
    """
    if torch.is_grad_enabled():
        q, k, v = qkv.unbind(1)
        q, k = q_norm(q), k_norm(k)

        if (rope is not None) and (pos is not None):
            q = rope(q.unsqueeze(0).transpose(1, 2), pos).transpose(1, 2).squeeze(0).to(dtype=run_dtype)
            k = rope(k.unsqueeze(0).transpose(1, 2), pos).transpose(1, 2).squeeze(0).to(dtype=run_dtype)

        return torch.stack([q, k, v], dim=1)

    else:
        qkv[:, 0].copy_(q_norm(qkv[:, 0]))
        qkv[:, 1].copy_(k_norm(qkv[:, 1]))

        if (rope is not None) and (pos is not None):
            qkv[:, 0].copy_(rope(qkv[:, 0].unsqueeze(0).transpose(1, 2), pos).transpose(1, 2).squeeze(0).to(dtype=run_dtype))
            qkv[:, 1].copy_(rope(qkv[:, 1].unsqueeze(0).transpose(1, 2), pos).transpose(1, 2).squeeze(0).to(dtype=run_dtype))

        return qkv
