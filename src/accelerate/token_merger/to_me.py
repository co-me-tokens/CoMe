"""
ToMe token merger — bipartite soft matching based token merging.

Implements ITokenMerger using the Token Merging (ToMe) algorithm:
cosine-similarity bipartite matching on even/odd token splits,
merging the top-r most similar pairs via scatter_reduce("mean").
"""

from dataclasses import dataclass

import torch
import jaxtyping as jt

from typing import Any

from ...interface.token_merger import ITokenMerger


class ToMe_TokenMerger(ITokenMerger):
    """Token Merging (ToMe) via bipartite soft matching, adapted to ITokenMerger.

    Args:
        start_idx: Number of leading special tokens excluded from merging.
        merge_ratio: Fraction of mergeable tokens to merge (0 to 1).
            0 = no merging, 1 = merge maximum possible (50% of spatial tokens).
    """

    @dataclass
    class _Stub:
        """Matching indices produced by a merge call, consumed by split."""
        r: int
        t: int
        unm_idx: jt.Int[torch.Tensor, "BS U 1"]
        src_idx: jt.Int[torch.Tensor, "BS R 1"]
        dst_idx: jt.Int[torch.Tensor, "BS R 1"]
        shape_bsnc: tuple[int, int, int, int]

    def __init__(self, start_idx: int, merge_ratio: float):
        if not 0 <= merge_ratio <= 1:
            raise ValueError(f"merge_ratio must be in [0, 1], got {merge_ratio}")
        self.start_idx = start_idx
        self.merge_ratio = merge_ratio

    @torch.no_grad()
    def build_ctx(self, feature: jt.Float[torch.Tensor, "B S C H W"]) -> Any:
        B, S, C, H, W = feature.shape
        return (B, S, C, H, W)

    @torch.no_grad()
    def merge(
        self,
        ctx: Any,
        tokens: jt.Float[torch.Tensor, "B S N C"],
        stub: _Stub | None = None,
        *,
        start_index: int | None = None,
    ) -> tuple[ITokenMerger.JaggedTokens, _Stub]:
        B, S, N, C = tokens.shape
        BS = B * S
        si = start_index if start_index is not None else self.start_idx
        flat = tokens.flatten(0, 1)                         # [BS, N, C]
        special = flat[:, :si, :]                            # [BS, si, C]
        spatial = flat[:, si:, :]                             # [BS, T, C]

        if stub is None:
            stub = self._compute_matching(spatial, (B, S, N, C))

        r = stub.r

        if r == 0:
            offsets = torch.arange(BS + 1, device=flat.device, dtype=torch.long) * N
            weight = torch.ones(BS * N, device=flat.device, dtype=torch.float)
            jagged = ITokenMerger.JaggedTokens(
                tokens=flat.reshape(1, BS * N, C), weight=weight, offset=offsets,
            )
            return jagged, stub

        src, dst = spatial[..., ::2, :], spatial[..., 1::2, :]
        t1 = src.shape[1]

        unm = src.gather(dim=-2, index=stub.unm_idx.expand(BS, t1 - r, C))
        src_gathered = src.gather(dim=-2, index=stub.src_idx.expand(BS, r, C))
        dst = dst.scatter_reduce(-2, stub.dst_idx.expand(BS, r, C), src_gathered, reduce="mean")

        merged = torch.cat([special, unm, dst], dim=1)       # [BS, M, C]
        M = merged.shape[1]

        offsets = torch.arange(BS + 1, device=flat.device, dtype=torch.long) * M

        w_special = torch.ones(BS, si, device=flat.device)
        w_unm = torch.ones(BS, t1 - r, device=flat.device)
        dst_count = torch.zeros(BS, dst.shape[1], device=flat.device)
        dst_count.scatter_add_(1, stub.dst_idx.squeeze(-1), torch.ones(BS, r, device=flat.device))
        w_dst = 1.0 + dst_count

        weight = torch.cat([w_special, w_unm, w_dst], dim=1).flatten()

        jagged = ITokenMerger.JaggedTokens(
            tokens=merged.reshape(1, BS * M, C), weight=weight, offset=offsets,
        )
        return jagged, stub

    @torch.no_grad()
    def split(
        self,
        ctx: Any,
        stub: _Stub,
        tokens: ITokenMerger.JaggedTokens,
    ) -> jt.Float[torch.Tensor, "B S N C"]:
        B, S, N, C = stub.shape_bsnc
        BS = B * S
        r = stub.r

        if r == 0:
            return tokens.tokens.reshape(B, S, N, C)

        t1 = (stub.t + 1) // 2
        unm_len = t1 - r
        M = self.start_idx + stub.t - r
        merged = tokens.tokens.reshape(BS, M, C)

        special = merged[:, :self.start_idx, :]
        unm = merged[:, self.start_idx:self.start_idx + unm_len, :]
        dst = merged[:, self.start_idx + unm_len:, :]

        src_from_dst = dst.gather(dim=-2, index=stub.dst_idx.expand(BS, r, C))

        spatial_out = torch.zeros(BS, stub.t, C, device=merged.device, dtype=merged.dtype)
        spatial_out[..., 1::2, :] = dst
        spatial_out.scatter_(dim=-2, index=(2 * stub.unm_idx).expand(BS, unm_len, C), src=unm)
        spatial_out.scatter_(dim=-2, index=(2 * stub.src_idx).expand(BS, r, C), src=src_from_dst)

        output = torch.cat([special, spatial_out], dim=1)
        return output.reshape(B, S, N, C)

    @torch.no_grad()
    def build_infer_mask(self, ctx: Any) -> jt.Bool[torch.Tensor, "B S H W"]:
        B, S, _C, H, W = ctx
        return torch.zeros(B, S, H, W, dtype=torch.bool)

    # ==== Internal ====

    def _compute_matching(
        self,
        spatial: jt.Float[torch.Tensor, "BS T C"],
        shape_bsnc: tuple[int, int, int, int],
    ) -> _Stub:
        BS, t, _C = spatial.shape
        max_r = t // 2
        r = int(max_r * self.merge_ratio)

        if r == 0:
            return self._Stub(
                r=0, t=t,
                unm_idx=torch.empty(0, device=spatial.device, dtype=torch.long),
                src_idx=torch.empty(0, device=spatial.device, dtype=torch.long),
                dst_idx=torch.empty(0, device=spatial.device, dtype=torch.long),
                shape_bsnc=shape_bsnc,
            )

        metric = spatial / spatial.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        return self._Stub(
            r=r, t=t,
            unm_idx=unm_idx, src_idx=src_idx, dst_idx=dst_idx,
            shape_bsnc=shape_bsnc,
        )
