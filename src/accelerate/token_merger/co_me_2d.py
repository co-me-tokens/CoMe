"""
CoMe 2D token merger — merges tokens in 2D spatial blocks of (group_h x group_w).

Provides a naive pure-PyTorch implementation and a CUDA-accelerated version
that reuses the existing 1D merge kernel via token rearrangement.
"""

import json
import torch
import torch.cuda.nvtx as nvtx
import jaxtyping as jt

from enum import Enum
from pathlib import Path
from typing import Any

from ...interface.token_merger import ITokenMerger
from ...utility.diagnostic import Diagnostics

from ..common.confidence_predictor import GlobalAttentionConfidence
from .co_me import func_token_merge, func_token_split


# ==== Helpers ====

def _build_2d_block_perm(
    H: int, W: int,
    group_h: int, group_w: int,
    device: torch.device | str,
) -> jt.Int[torch.Tensor, "H*W"]:
    """Return a permutation that reorders row-major image positions
    so that 2D blocks of ``(group_h, group_w)`` become contiguous."""
    M_h, M_w = H // group_h, W // group_w
    indices = torch.arange(H * W, device=device, dtype=torch.int64).reshape(M_h, group_h, M_w, group_w)
    return indices.permute(0, 2, 1, 3).reshape(-1)


def _validate_token_merge_2d_args(
    tokens: torch.Tensor,
    start_index: int,
    group_h: int,
    group_w: int,
    H: int,
    W: int,
    merge_mask: torch.Tensor,
) -> None:
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if group_h <= 0 or group_w <= 0:
        raise ValueError(f"group_h and group_w must be > 0, got ({group_h}, {group_w})")
    if H % group_h != 0:
        raise ValueError(f"H ({H}) must be divisible by group_h ({group_h})")
    if W % group_w != 0:
        raise ValueError(f"W ({W}) must be divisible by group_w ({group_w})")
    if merge_mask.dtype != torch.bool:
        raise ValueError(f"merge_mask must be of dtype torch.bool, got {merge_mask.dtype}")

    B, N, _C = tokens.shape
    _Bm, M = merge_mask.shape
    M_expected = (H // group_h) * (W // group_w)

    if _Bm != B:
        raise ValueError(f"merge_mask batch ({_Bm}) must match tokens batch ({B})")
    if N != start_index + H * W:
        raise ValueError(
            f"Expected N = start_index + H * W = {start_index} + {H * W} = "
            f"{start_index + H * W}, got N = {N}"
        )
    if M != M_expected:
        raise ValueError(
            f"Expected M = (H // group_h) * (W // group_w) = {M_expected}, got M = {M}"
        )


# ==== Naive (pure-PyTorch) 2D merge ====

def naive_func_token_merge_2d(
    tokens: jt.Float[torch.Tensor, "B N C"],
    start_index: int,
    group_h: int,
    group_w: int,
    H: int,
    W: int,
    merge_mask: jt.Bool[torch.Tensor, "B M"],
) -> tuple[ITokenMerger.JaggedTokens, jt.Int[torch.Tensor, "B*N"]]:
    """Merge 2D spatial blocks according to *merge_mask* (pure PyTorch).

    Args:
        tokens:      ``[B, N, C]`` tensor where ``N = start_index + H * W``.
        start_index: Number of leading special tokens excluded from merging.
        group_h:     Block height.
        group_w:     Block width.
        H, W:        Spatial dimensions of the image token grid.
        merge_mask:  ``[B, M]`` bool where ``M = (H // group_h) * (W // group_w)``.
                     ``True`` → merge the block into one token (mean),
                     ``False`` → keep all ``group_h * group_w`` tokens.

    Returns:
        ``(JaggedTokens, rev_id)`` where ``rev_id[B*N]`` maps every original
        token position (row-major) to its position in the merged flat output.
    """
    _validate_token_merge_2d_args(tokens, start_index, group_h, group_w, H, W, merge_mask)

    device  = tokens.device
    B, N, C = tokens.shape
    M_h, M_w = H // group_h, W // group_w
    M = M_h * M_w
    G = group_h * group_w

    # Per-batch output lengths
    k_b = merge_mask.sum(dim=1, dtype=torch.int64)
    L_b = start_index + (M - k_b) * G + k_b

    offsets = torch.empty((B + 1,), device=device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(L_b, dim=0)
    P = int(offsets[-1].item())

    out    = torch.empty((P, C), device=device, dtype=tokens.dtype)
    rev    = torch.empty((B, N), device=device, dtype=torch.int64)
    weight = torch.empty((P,),   device=device, dtype=torch.float)
    base   = offsets[:-1]

    # 1. Special tokens
    if start_index > 0:
        special_offset = torch.arange(start_index, device=device, dtype=torch.int64)
        special_pos = base[:, None] + special_offset[None, :]
        out.index_copy_(0, special_pos.reshape(-1), tokens[:, :start_index, :].reshape(-1, C))
        weight[special_pos.reshape(-1)] = 1.0
        rev[:, :start_index] = special_pos

    # 2. Reshape image tokens into 2D blocks: [B, M, G, C]
    img = tokens[:, start_index:, :].reshape(B, H, W, C)
    img_blocks = img.reshape(B, M_h, group_h, M_w, group_w, C)
    img_blocks = img_blocks.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, M, G, C)

    merged_means = img_blocks.mean(dim=2)

    # 3. Group destinations (same logic as 1D)
    group_len    = torch.where(merge_mask, 1, G)
    group_prefix = torch.cumsum(group_len, dim=1) - group_len
    group_base   = base[:, None] + start_index + group_prefix

    token_offset = torch.arange(G, device=device, dtype=torch.int64)
    unmerged_pos = group_base[..., None] + token_offset   # [B, M, G]

    # 4. Scatter merged groups
    merged_pos = group_base[merge_mask]
    out.index_copy_(0, merged_pos, merged_means[merge_mask])
    weight[merged_pos] = float(G)

    # 5. Scatter unmerged groups
    unmerged_mask     = ~merge_mask
    unmerged_pos_flat = unmerged_pos[unmerged_mask].reshape(-1)
    out.index_copy_(0, unmerged_pos_flat, img_blocks[unmerged_mask].reshape(-1, C))
    weight[unmerged_pos_flat] = 1.0

    # 6. Build rev_id in block order, then scatter to original row-major order
    block_rev = torch.where(merge_mask[..., None], group_base[..., None], unmerged_pos)

    hw_indices = torch.arange(H * W, device=device, dtype=torch.int64).reshape(M_h, group_h, M_w, group_w)
    hw_indices = hw_indices.permute(0, 2, 1, 3).reshape(M * G)

    rev_img = torch.empty((B, H * W), device=device, dtype=torch.int64)
    rev_img.scatter_(1, hw_indices.unsqueeze(0).expand(B, -1), block_rev.reshape(B, M * G))
    rev[:, start_index:] = rev_img

    jagged = ITokenMerger.JaggedTokens(tokens=out.unsqueeze(0), weight=weight, offset=offsets)
    return jagged, rev.flatten()


# ==== CUDA-accelerated 2D merge (rearrange + 1D kernel) ====

def func_token_merge_2d(
    tokens: jt.Float[torch.Tensor, "B N C"],
    start_index: int,
    group_h: int,
    group_w: int,
    H: int,
    W: int,
    merge_mask: jt.Bool[torch.Tensor, "B M"],
) -> tuple[ITokenMerger.JaggedTokens, jt.Int[torch.Tensor, "B*N"]]:
    """Merge 2D spatial blocks via rearrangement + existing 1D merge kernel.

    Same semantics as :func:`naive_func_token_merge_2d` but dispatches to
    the CUDA-accelerated 1D merge path when available.
    """
    _validate_token_merge_2d_args(tokens, start_index, group_h, group_w, H, W, merge_mask)

    B, N, C = tokens.shape
    G = group_h * group_w

    img_perm = _build_2d_block_perm(H, W, group_h, group_w, tokens.device)

    full_perm = torch.cat([
        torch.arange(start_index, device=tokens.device, dtype=torch.int64),
        img_perm + start_index,
    ])

    rearranged = tokens[:, full_perm, :].contiguous()

    jagged, rev_id_rearranged = func_token_merge(rearranged, start_index, G, merge_mask)

    # Remap rev_id from rearranged order back to original row-major order
    inv_full_perm = torch.empty(N, device=tokens.device, dtype=torch.int64)
    inv_full_perm[full_perm] = torch.arange(N, device=tokens.device, dtype=torch.int64)

    rev_rearranged = rev_id_rearranged.reshape(B, N)
    rev_original   = rev_rearranged.gather(1, inv_full_perm.unsqueeze(0).expand(B, -1))

    return jagged, rev_original.flatten()


# ==== ITokenMerger subclasses ====

class _CoMe_2D_Base(ITokenMerger):
    """Shared base for 2D token mergers (confidence predictor + build_ctx + split + build_infer_mask)."""

    class ConfidenceMode(Enum):
        Ranking          = "pairwise-rank"
        Absolute_Score   = "abs_mse"
        Normalized_Score = "mse"

    def __init__(
        self,
        start_idx: int,
        group_h: int,
        group_w: int,
        threshold: float,
        ckpt_path: str | Path,
        device: str | torch.device,
    ):
        self.start_idx = start_idx
        self.group_h   = group_h
        self.group_w   = group_w
        self.threshold = threshold

        config_path = Path(ckpt_path, "config.json")
        weight_path = Path(ckpt_path, "checkpoint.pth")
        assert config_path.exists() and weight_path.exists()

        with open(config_path, "r") as f:
            config = json.load(f)
        predictor = GlobalAttentionConfidence(
            hidden_dims=config["predictor_dims"],
            activation=config["predictor_active"],
            num_register=config.get("predictor_num_register", 0),
        )
        predictor.load_state_dict(torch.load(weight_path, weights_only=True))
        self.predictor = predictor.to(device=device, dtype=torch.bfloat16).eval()
        self.conf_mode = self.ConfidenceMode(config["loss_function"])

    def build_ctx(self, feature: jt.Float[torch.Tensor, "B S C H W"]) -> Any:
        B, S, C, H, W = feature.shape
        if H % self.group_h != 0:
            raise ValueError(f"H ({H}) must be divisible by group_h ({self.group_h})")
        if W % self.group_w != 0:
            raise ValueError(f"W ({W}) must be divisible by group_w ({self.group_w})")

        M_h, M_w = H // self.group_h, W // self.group_w
        G = self.group_h * self.group_w

        Diagnostics.log(f"build_ctx: B={B}, S={S}, C={C}, H={H}, W={W}, seqlen_for_predictor={B}x{S*H*W}={B*S*H*W}")
        with nvtx.range("confidence_predictor"):
            confidence: torch.Tensor = self.predictor(
                feature.permute(0, 1, 3, 4, 2).bfloat16(), BPHWC=(B, S, H, W, C)
            ).squeeze(-1)

        # Reshape per-token confidence into 2D blocks and average
        conf_2d     = confidence.reshape(B * S, H, W)
        conf_blocks = conf_2d.reshape(B * S, M_h, self.group_h, M_w, self.group_w)
        conf_blocks = conf_blocks.permute(0, 1, 3, 2, 4).reshape(B * S, M_h * M_w, G)

        match self.conf_mode:
            case self.ConfidenceMode.Normalized_Score:
                flat_conf = confidence.reshape(B, -1)
                norm_conf = (flat_conf - flat_conf.mean(dim=-1, keepdim=True)) / (
                    flat_conf.std(dim=-1, keepdim=True) + 1e-3
                )
                norm_2d     = norm_conf.reshape(B * S, H, W)
                norm_blocks = norm_2d.reshape(B * S, M_h, self.group_h, M_w, self.group_w)
                norm_blocks = norm_blocks.permute(0, 1, 3, 2, 4).reshape(B * S, M_h * M_w, G)
                mask = norm_blocks.mean(dim=-1) < self.threshold

            case self.ConfidenceMode.Ranking:
                if not (0.0 <= self.threshold <= 1.0):
                    raise ValueError(
                        f"Ranking mode requires threshold in [0, 1], got {self.threshold}"
                    )
                
                block_means = conf_blocks.mean(dim=-1).reshape(B, S * M_h * M_w)
                percentile_val = torch.quantile(
                    block_means.float(), self.threshold, dim=-1, keepdim=True
                )
                mask = (block_means < percentile_val).reshape(B * S, M_h * M_w)

            case self.ConfidenceMode.Absolute_Score:
                mask = conf_blocks.mean(dim=-1) < self.threshold

        if Diagnostics.is_active():
            orig_token_cnt = B * S * H * W
            comp_token_cnt = mask.sum() + (~mask).sum() * G + (B * S) * self.start_idx
            mask_ratio = mask.float().mean().item() * 100.0
            compression_ratio = (1 - (comp_token_cnt / orig_token_cnt)) * 100.0
            Diagnostics.log(
                f"Mask generation report (2D):\n"
                f"\tinput token count: {orig_token_cnt}\n"
                f"\tcompressed       : {comp_token_cnt}\n"
                f"\tcompress ratio   : {compression_ratio:.2f}%\n"
                f"\tmask ratio       : {mask_ratio:.2f}%"
            )

        return (mask, (B, S, C, H, W))

    def split(self, ctx: Any, stub: Any, tokens: ITokenMerger.JaggedTokens) -> jt.Float[torch.Tensor, "B S N C"]:
        rev_id, shape_bsnc = stub
        B, S, N, C = shape_bsnc
        split_tokens = func_token_split(tokens=tokens, rev_id=rev_id, bsnc_shape=shape_bsnc)
        return split_tokens.view(B, S, N, C)

    def build_infer_mask(self, ctx: Any) -> jt.Bool[torch.Tensor, "B S H W"]:
        mask, (B, S, _, H, W) = ctx
        M_h, M_w = H // self.group_h, W // self.group_w
        infer_mask = mask.reshape(B * S, M_h, M_w)
        infer_mask = infer_mask[:, :, None, :, None].expand(
            -1, -1, self.group_h, -1, self.group_w
        )
        return infer_mask.reshape(B, S, H, W)


class Naive_CoMe_2D_TokenMerger(_CoMe_2D_Base):
    """2D token merger using pure-PyTorch merge (no CUDA kernels)."""

    def merge(self, ctx: Any, tokens: jt.Float[torch.Tensor, "B S N C"], stub: Any | None = None, *, start_index: int | None = None) -> tuple[ITokenMerger.JaggedTokens, Any]:
        mask, (B, S, C, H, W) = ctx
        si = start_index if start_index is not None else self.start_idx
        merged_tokens, rev_id = naive_func_token_merge_2d(
            tokens.flatten(0, 1), si, self.group_h, self.group_w, H, W, mask,
        )
        return merged_tokens, (rev_id, tokens.shape)


class CoMe_2D_TokenMerger(_CoMe_2D_Base):
    """2D token merger using CUDA-accelerated 1D kernel via token rearrangement."""

    def merge(self, ctx: Any, tokens: jt.Float[torch.Tensor, "B S N C"], stub: Any | None = None, *, start_index: int | None = None) -> tuple[ITokenMerger.JaggedTokens, Any]:
        mask, (B, S, C, H, W) = ctx
        si = start_index if start_index is not None else self.start_idx
        merged_tokens, rev_id = func_token_merge_2d(
            tokens.flatten(0, 1), si, self.group_h, self.group_w, H, W, mask,
        )
        return merged_tokens, (rev_id, tokens.shape)
