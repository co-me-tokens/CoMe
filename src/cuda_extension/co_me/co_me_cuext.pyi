"""
CUDA kernels for Co-Me project
"""
from __future__ import annotations
import torch
import typing
__all__: list[str] = ['co_me_token_merge_cuda', 'co_me_token_merge_cuda_legacy', 'co_me_token_merge_scan_legacy', 'co_me_token_split_cuda']
def co_me_token_merge_cuda(arg0: torch.Tensor, arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused token merge: scan + scatter + rev_id.
    
    Args:
        tokens (torch.Tensor): [B, N, C] float tensor on CUDA.
        start_index (int): number of special prefix tokens per sample.
        group_size (int): G, tokens per merge group.
        merge_mask (torch.Tensor): [B, M] bool tensor on CUDA.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            out_flat [P, C], offsets [B+1], rev_id [B*N], weight [P].
    """
def co_me_token_merge_cuda_legacy(arg0: typing.SupportsInt, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor | None) -> torch.Tensor:
    """
    Contract and merge groups of tokens with optional averaging.
    
    Args:
        prefix_cnt (int): number of prefix token slots to reserve.
        tokens (torch.Tensor): [B, N, G, C] typed tensor (float32, float16, bfloat16, or int64).
        dst_lidx (torch.Tensor): [B, N * G] int64 tensor of destination indices.
        out_ptr (torch.Tensor | None): [B, R, C] shaped tensor, if provided will write result into this pointer.
    
    Returns:
        torch.Tensor: [B, prefix_cnt + M, C] output tensor after contraction.
    """
def co_me_token_merge_scan_legacy(arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: torch.Tensor) -> torch.Tensor:
    """
    Scan and group-contract a boolean mask per batch.
    
    Args:
        prefix_cnt (int64_t): number of prefix token slots to reserve.
        G (int64_t): group size for repeat and contraction.
        mask (torch::Tensor): [B, N] boolean (or uint8) tensor on CUDA.
    
    Returns:
        torch::Tensor: [B, N * G] int64 tensor of destination indices (with prefix offset applied).
    """
def co_me_token_split_cuda(arg0: torch.Tensor, arg1: torch.Tensor, arg2: typing.SupportsInt) -> torch.Tensor:
    """
    Token split (inverse of merge): gather by rev_id.
    
    Args:
        merged_flat (torch.Tensor): [P, C] float tensor on CUDA.
        rev_id (torch.Tensor): [total] int64 gather indices.
        C (int): feature dimension.
    
    Returns:
        torch.Tensor: [total, C] restored tokens.
    """
