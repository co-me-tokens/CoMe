#pragma once
#include <ATen/ATen.h>
#include <tuple>

/**
 * Fused token merge: scan + scatter + rev_id in a single host call.
 *
 * @param tokens      [B, N, C] contiguous float tensor.
 * @param start_index Number of special (non-group) prefix tokens per sample.
 * @param group_size  G – number of tokens per merge group.
 * @param merge_mask  [B, M] bool tensor.  N == start_index + M * G.
 *
 * @return (out_flat, offsets, rev_id)
 *   out_flat  [P, C] – jagged merged tokens (P = sum of per-batch lengths).
 *   offsets   [B+1]  – cumulative batch boundaries into out_flat.
 *   rev_id    [B*N]  – reverse index: original-token → merged-token position.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
co_me_token_merge_cuda(
    at::Tensor  tokens,
    int64_t     start_index,
    int64_t     group_size,
    at::Tensor  merge_mask);

/**
 * Token split (inverse of merge): gathers merged tokens back to original layout.
 *
 * @param merged_flat [P, C] contiguous float tensor.
 * @param rev_id      [total] int64 tensor of gather indices.
 * @param C           Feature dimension (must match merged_flat.size(1)).
 *
 * @return restored [total, C].
 */
at::Tensor
co_me_token_split_cuda(
    at::Tensor  merged_flat,
    at::Tensor  rev_id,
    int64_t     C);
