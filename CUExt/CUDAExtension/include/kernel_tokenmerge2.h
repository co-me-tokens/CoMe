#pragma once
#include <ATen/ATen.h>
#include <c10/util/Optional.h>

/**
 * @param prefix_cnt: The number of prefixes token to retain space for before the contracted tokens.
 * 
 * @param tokens    : (float32, bfloat16, float16, int64) typed torch.Tensor of shape [B, N, G, C].
 *                    B - Batch size
 *                    N - Number of groups per sample
 *                    G - Number of tokens per group
 *                    C - Number of features per token
 *                    The typical range of these values are: B ~ [1, 64], N ~ [1024, 4096], G ~ [4 or 8], C ~[1024, 2048]
 * 
 * @param dst_lidx  : (int64) typed torch.Tensor of shape [B, N * G].
 *                    This is the precomputed index of each tokens after the contraction. For each of the G consecutive values
 *                    in dst_lidx, there are only two possible cases:
 *                          * Case A - all G values are increasing by G compared to the previous value.
 *                            (idx[b, i+1] = idx[b, i] + G)
 *                          * Case B - all G values are increasing by 1 compared to the previous value.
 *                            (idx[b, i+1] = idx[b, i] + 1)
 *                    it is also guaranteed that idx[b, k*G] must be the multiple of G in this case.
 *                    
 *                    In Case A, all the kernel will do is to copy value from x[b, k, i] to out[b, floor(idx[k * G + i] / G)] for each 
 *                    i in [0, G]
 * 
 *                    In Case B, since all G values in idx[b, k*G : (k+1)*G] will point to same value after / G operation. 
 *                    the kernel will follow this logic
 *                          out[b, idx[k * G] / G] = x[b, k].mean(dim=-1, keepdim=True)
 * 
 */
at::Tensor token_grpcontract_v2_merge(const int64_t prefix_cnt, at::Tensor tokens, at::Tensor dst_lidx, c10::optional<at::Tensor> out_ptr);


/**
 * Given a batch of boolean mask of shape [B, N]
 * 
 * For each mask, do:
 *      0. mask     = repeat_interleave(mask, G)
 *      1. score    = map(lambda bool: bool ? 1 : G, mask)
 *      2. dst_fidx = exclusive_scan(score)
 *      3. dst_lidx = dst_fidx // G
 *      4. Return the contraction index list of shape [1, N * G]
 * 
 * Concatenate all contraction masks to retrieve a [B, N*G] shaped contraction index
 */
at::Tensor token_grpcontract_v2_scan (const int64_t prefix_cnt, const int64_t G, at::Tensor mask);
