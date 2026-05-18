"""
CoMe token merger — CUDA-accelerated token merge / split.
"""

import json
import torch
import jaxtyping as jt

from enum import Enum
from pathlib import Path
from typing import Any, cast

from ...interface.token_merger import ITokenMerger
from ...cuda_extension import co_me_cuext
from ...utility.feature import CUDAExtension
from ...utility.diagnostic import Diagnostics

from ..common.confidence_predictor import GlobalAttentionConfidence


class CoMe_TokenMerger(ITokenMerger):
    
    class ConfidenceMode(Enum):
        Ranking          = "pairwise-rank"
        Absolute_Score   = "abs_mse"
        Normalized_Score = "mse"
    
    def __init__(self, start_idx: int, group_size: int, threshold: float, ckpt_path: str | Path, device: str | torch.device):
        self.start_idx  = start_idx
        self.group_size = group_size
        self.threshold  = threshold

        config_path = Path(ckpt_path, "config.json")
        weight_path = Path(ckpt_path, "checkpoint.pth")
        assert (config_path.exists() and weight_path.exists())
        
        with open(config_path, "r") as f: config = json.load(f)
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
        assert ((H * W) % self.group_size) == 0

        confidence = self.predictor(feature.permute(0, 1, 3, 4, 2).bfloat16(), BPHWC=(B, S, H, W, C)).squeeze(-1)
        
        match self.conf_mode:
            case self.ConfidenceMode.Normalized_Score:
                confidence = confidence.reshape(B, -1)  # Per-batch normalize, not per-sample normalize.
                norm_conf  = (confidence - confidence.mean(dim=-1, keepdim=True)) / (confidence.std(dim=-1, keepdim=True) + 1e-3)
                mask = norm_conf.reshape(B*S, H*W//self.group_size, self.group_size).mean(dim=-1)
                mask = mask < self.threshold
            
            case self.ConfidenceMode.Ranking:
                if not (0.0 <= self.threshold <= 1.0):
                    raise ValueError(
                        f"Ranking mode requires threshold in [0, 1], got {self.threshold}"
                    )
                group_means = confidence.reshape(B*S, H*W//self.group_size, self.group_size).mean(dim=-1)
                percentile_val = torch.quantile(
                    group_means.float(), self.threshold, dim=-1, keepdim=True
                )
                mask = group_means < percentile_val
            
            case self.ConfidenceMode.Absolute_Score:
                mask = confidence.reshape(B*S, H*W//self.group_size, self.group_size).mean(dim=-1)
                mask = mask < self.threshold
        
        if Diagnostics.is_active():
            orig_token_cnt = B * S * H * W
            comp_token_cnt = mask.sum() + (~mask).sum() * self.group_size + (B * S) * self.start_idx
            mask_ratio     = mask.float().mean().item() * 100.
            compression_ratio = (1 - (comp_token_cnt / orig_token_cnt)) * 100.
            Diagnostics.log(f"Mask generation report:\n"
                            f"\tinput token count: {orig_token_cnt}\n"
                            f"\tcompressed       : {comp_token_cnt}\n"
                            f"\tcompress ratio   : {compression_ratio:.2f}%\n"
                            f"\tmask ratio       : {mask_ratio:.2f}%")
        
        return (mask, (B, S, C, H, W))

    def merge(self, ctx, tokens: jt.Float[torch.Tensor, "B S N C"], stub: Any | None = None, *, start_index: int | None = None) -> tuple[ITokenMerger.JaggedTokens, Any]:
        mask, shape = ctx
        si = start_index if start_index is not None else self.start_idx
        merged_tokens, rev_id = func_token_merge(tokens.flatten(0, 1), si, self.group_size, mask)
        shape_bsnc = tokens.shape
        return merged_tokens, (rev_id, shape_bsnc)

    def split(self, ctx, stub: Any, tokens: ITokenMerger.JaggedTokens) -> jt.Float[torch.Tensor, "B S N C"]:
        rev_id, shape_bsnc = stub
        B, S, N, C = shape_bsnc
        split_tokens = func_token_split(tokens=tokens, rev_id=rev_id, bsnc_shape=shape_bsnc)
        return split_tokens.view(B, S, N, C)

    def build_infer_mask(self, ctx) -> jt.Bool[torch.Tensor, "B S H W"]:
        mask, shape_bschw = ctx
        B, S, _, H, W = shape_bschw        
        infer_mask = mask.unsqueeze(-1).repeat(1, 1, self.group_size).flatten(-2, -1)
        return infer_mask.reshape(B, S, H, W)


def _validate_token_merge_args(
    tokens: torch.Tensor,
    start_index: int,
    group_size: int,
    merge_mask: torch.Tensor,
) -> None:
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if merge_mask.dtype != torch.bool:
        raise ValueError(f"merge_mask must be of dtype torch.bool, got {merge_mask.dtype}")
    B, N, _C = tokens.shape
    _Bm, M = merge_mask.shape
    if _Bm != B:
        raise ValueError(f"merge_mask batch ({_Bm}) must match tokens batch ({B})")
    if N != start_index + M * group_size:
        raise ValueError(
            f"Expected N = start_index + M * group_size = {start_index} + {M} * {group_size} = "
            f"{start_index + M * group_size}, got N = {N}"
        )


def co_me_token_merge_cuda_entrypoint(
    tokens: jt.Float[torch.Tensor, "B N C"],
    start_index: int,
    group_size: int,
    merge_mask: jt.Bool[torch.Tensor, "B M"]
) -> tuple[ITokenMerger.JaggedTokens, jt.Int[torch.Tensor, "B*N"]]:
    _validate_token_merge_args(tokens, start_index, group_size, merge_mask)
    cuda_result = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        co_me_cuext.co_me_token_merge_cuda(tokens.contiguous(), start_index, group_size, merge_mask.contiguous()),
    )
    out_flat, offsets, rev_id, weight = cuda_result
    jagged = ITokenMerger.JaggedTokens(
        tokens=out_flat.unsqueeze(0), weight=weight, offset=offsets.long()
    )
    return jagged, rev_id


setattr(co_me_cuext, "co_me_token_merge_cuda_entrypoint", co_me_token_merge_cuda_entrypoint)


@CUDAExtension.accelerate("co_me_token_merge_cuda_entrypoint")
def func_token_merge(
    tokens: jt.Float[torch.Tensor, "B N C"],
    start_index: int,
    group_size: int,
    merge_mask: jt.Bool[torch.Tensor, "B M"],
) -> tuple[ITokenMerger.JaggedTokens, jt.Int[torch.Tensor, "B*N"]]:
    """Merge token groups according to *merge_mask*.

    Args:
        tokens:      ``[B, N, C]`` contiguous CUDA tensor.
        start_index: Number of leading special tokens excluded from merging.
        group_size:  ``G`` — tokens per merge group.
        merge_mask:  ``[B, M]`` bool — ``True`` → merge the group into one
                     token (mean of ``G``), ``False`` → keep ``G`` tokens.

    Returns:
        ``(JaggedTokens, rev_id)`` where ``rev_id[B*N]`` maps every original
        token to its position in the merged flat output.
    """
    _validate_token_merge_args(tokens, start_index, group_size, merge_mask)
    device     = tokens.device
    B, N, C    = tokens.shape
    _Bm, M    = merge_mask.shape

    k_b = merge_mask.sum(dim=1, dtype=torch.int64)
    L_b = start_index + (M - k_b) * group_size + k_b

    offsets = torch.empty((B + 1,), device=device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(L_b, dim=0)
    P = int(offsets[-1].item())

    out = torch.empty((P, C), device=device, dtype=tokens.dtype)
    rev = torch.empty((B, N), device=device, dtype=torch.int64)
    base = offsets[:-1]

    special_offset = torch.arange(start_index, device=device, dtype=torch.int64)
    special_pos = base[:, None] + special_offset[None, :]
    special_tok = tokens[:, :start_index, :]

    out.index_copy_(0, special_pos.reshape(-1), special_tok.reshape(-1, C))
    rev[:, :start_index] = special_pos

    img = tokens[:, start_index:, :].reshape(B, M, group_size, C)
    merged = img.mean(dim=2)

    group_len    = torch.where(merge_mask, 1, group_size)
    group_prefix = torch.cumsum(group_len, dim=1) - group_len
    group_base   = base[:, None] + start_index + group_prefix

    token_offset = torch.arange(group_size, device=device, dtype=torch.int64)
    unmerged_pos = group_base[..., None] + token_offset

    merged_pos = group_base[merge_mask]
    out.index_copy_(0, merged_pos, merged[merge_mask])

    unmerged_mask = ~merge_mask
    unmerged_pos_flat = unmerged_pos[unmerged_mask].reshape(-1)
    out.index_copy_(0, unmerged_pos_flat, img[unmerged_mask].reshape(-1, C))

    img_rev = torch.where(merge_mask[..., None], group_base[..., None], unmerged_pos)
    rev[:, start_index:] = img_rev.reshape(B, M * group_size)

    weight = torch.empty((P,), device=device, dtype=torch.float)
    weight[special_pos.reshape(-1)] = 1.0
    weight[merged_pos] = float(group_size)
    weight[unmerged_pos_flat] = 1.0

    jagged = ITokenMerger.JaggedTokens(tokens=out.unsqueeze(0), weight=weight, offset=offsets)
    return jagged, rev.flatten()


def _token_split_cuda(
    tokens: ITokenMerger.JaggedTokens,
    rev_id: jt.Int[torch.Tensor, "B*N"],
    bsnc_shape: tuple[int, int, int, int]
) -> jt.Float[torch.Tensor, "B N C"]:
    B, S, N, C = bsnc_shape
    flattened_tokens = tokens.tokens.squeeze(0)
    restored = co_me_cuext.co_me_token_split_cuda(
        flattened_tokens.contiguous(), rev_id.contiguous(), C
    )
    return restored.reshape(B * S, N, C)


setattr(co_me_cuext, "token_split_py_wrapper", _token_split_cuda)


@CUDAExtension.accelerate("token_split_py_wrapper")
def func_token_split(
    tokens: ITokenMerger.JaggedTokens,
    rev_id: jt.Int[torch.Tensor, "B*N"],
    bsnc_shape: tuple[int, int, int, int]
) -> jt.Float[torch.Tensor, "B N C"]:
    """
    Inverse of :func:`token_merge` — gather merged tokens back to the
    original ``[B*S, N, C]`` layout using *rev_id*.
    """
    B, S, N, C = bsnc_shape
    flattened_tokens = tokens.tokens.squeeze(0)
    restored = flattened_tokens.index_select(0, rev_id)
    return restored.reshape(B * S, N, C)
