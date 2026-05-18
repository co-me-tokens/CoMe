from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
from dataclasses import dataclass

import torch
import jaxtyping as jt


class ITokenMerger(ABC):
    @dataclass
    class JaggedTokens:
        tokens: jt.Float[torch.Tensor, "1 N C"]
        weight: jt.Int[torch.Tensor, "N"]
        offset: jt.Int[torch.Tensor, "M"]
        
        def apply(self, func: Callable[[torch.Tensor], torch.Tensor]) -> ITokenMerger.JaggedTokens:
            return ITokenMerger.JaggedTokens(
                tokens=func(self.tokens), weight=self.weight, offset=self.offset
            )

        def add_suffix(self, suffix: jt.Float[torch.Tensor, "M T C"]) -> ITokenMerger.JaggedTokens:
            """Append ``T`` dense tokens to each of the ``M`` jagged segments.

            Args:
                suffix: ``(M, T, C)`` tensor where ``M = len(offset) - 1``.

            Returns:
                A new ``JaggedTokens`` with each segment extended by ``T`` tokens
                (weight 1 for every appended token).
            """
            M = self.offset.shape[0] - 1
            if suffix.shape[0] != M:
                raise ValueError(
                    f"suffix batch dim ({suffix.shape[0]}) must match segment count ({M})"
                )
            T, C = suffix.shape[1], suffix.shape[2]
            P_old = self.tokens.shape[1]
            P_new = P_old + M * T

            delta = torch.arange(M + 1, device=self.offset.device, dtype=self.offset.dtype) * T
            new_offset = self.offset + delta

            new_tokens = torch.empty(1, P_new, C, device=self.tokens.device, dtype=self.tokens.dtype)
            new_weight = torch.ones(P_new, device=self.weight.device, dtype=self.weight.dtype)

            old_range = torch.arange(P_old, device=self.offset.device)
            seg_ids = torch.searchsorted(self.offset[1:], old_range, right=True)
            new_pos = old_range + seg_ids * T
            new_tokens[0].index_copy_(0, new_pos, self.tokens[0])
            new_weight.index_copy_(0, new_pos, self.weight)

            suffix_starts = new_offset[1:] - T
            suffix_idx = (suffix_starts[:, None] + torch.arange(T, device=self.offset.device)).reshape(-1)
            new_tokens[0].index_copy_(0, suffix_idx, suffix.reshape(-1, C))

            return ITokenMerger.JaggedTokens(tokens=new_tokens, weight=new_weight, offset=new_offset)

        def split_suffix(self, count: int) -> tuple[ITokenMerger.JaggedTokens, jt.Float[torch.Tensor, "M T C"]]:
            """Remove the last ``count`` tokens from each segment (inverse of :meth:`add_suffix`).

            Args:
                count: Number of trailing tokens to strip from every segment.

            Returns:
                ``(stripped, suffix)`` where ``stripped`` is the shortened
                ``JaggedTokens`` and ``suffix`` is ``(M, count, C)``.
            """
            M = self.offset.shape[0] - 1
            T, C = count, self.tokens.shape[2]
            P_old = self.tokens.shape[1]
            P_new = P_old - M * T

            if P_new < 0:
                raise ValueError(
                    f"Cannot strip {T} tokens from each of {M} segments "
                    f"(total tokens = {P_old})"
                )

            delta = torch.arange(M + 1, device=self.offset.device, dtype=self.offset.dtype) * T
            orig_offset = self.offset - delta

            seg_lens = orig_offset[1:] - orig_offset[:-1]
            if (seg_lens < 0).any():
                raise ValueError(
                    f"count={T} exceeds the length of at least one segment"
                )

            suffix_starts = self.offset[1:] - T
            suffix_idx = (suffix_starts[:, None] + torch.arange(T, device=self.offset.device)).reshape(-1)
            suffix = self.tokens[0].index_select(0, suffix_idx).reshape(M, T, C)

            new_range = torch.arange(P_new, device=self.offset.device)
            seg_ids = torch.searchsorted(orig_offset[1:], new_range, right=True)
            orig_pos = new_range + seg_ids * T
            new_tokens = self.tokens[0].index_select(0, orig_pos).unsqueeze(0)
            new_weight = self.weight.index_select(0, orig_pos)

            return ITokenMerger.JaggedTokens(tokens=new_tokens, weight=new_weight, offset=orig_offset), suffix
    
    @abstractmethod
    def build_ctx(self, feature: jt.Float[torch.Tensor, "B S C H W"]) -> Any:
        ...
    
    @abstractmethod
    def merge(self, ctx, tokens: jt.Float[torch.Tensor, "B S N C"], stub: Any | None = None, *, start_index: int | None = None) -> tuple[JaggedTokens, Any]:
        ...
    
    @abstractmethod
    def split(self, ctx, stub: Any, tokens: JaggedTokens) -> jt.Float[torch.Tensor, "B S N C"]:
        ...

    @abstractmethod
    def build_infer_mask(self, ctx) -> jt.Bool[torch.Tensor, "B S H W"]:
        ...
