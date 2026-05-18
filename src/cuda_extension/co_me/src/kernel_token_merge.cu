#include <kernel_token_merge.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>


// ==== Block-level prefix scan: one block per batch row, tiles in shared memory ====
constexpr int SCAN_BLOCK = 256;

__global__ void block_prefix_scan_kernel(
    const bool* __restrict__ mask,           // [B, M]
    int         G,
    int         M,
    int         start_index,
    int*        __restrict__ group_prefix,   // [B, M]
    int*        __restrict__ L_b)            // [B]
{
    typedef cub::BlockScan<int, SCAN_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int b = blockIdx.x;
    const bool* row_mask   = mask + b * M;
    int*        row_prefix = group_prefix + b * M;

    int running_total = 0;

    for (int tile_start = 0; tile_start < M; tile_start += SCAN_BLOCK) {
        int idx = tile_start + threadIdx.x;
        int val = (idx < M) ? (row_mask[idx] ? 1 : G) : 0;

        int output, tile_aggregate;
        BlockScan(temp_storage).ExclusiveSum(val, output, tile_aggregate);
        __syncthreads();

        if (idx < M) {
            row_prefix[idx] = running_total + output;
        }
        running_total += tile_aggregate;
    }

    if (threadIdx.x == 0) {
        L_b[b] = start_index + running_total;
    }
}


// ==== Optimised scatter kernel (float4 vectorised, int32 indexing) ====
__global__ void token_merge_scatter_kernel_vec4(
    const float*  __restrict__ tokens,
    const bool*   __restrict__ mask,
    const int*    __restrict__ group_prefix,
    const int*    __restrict__ offsets,
    int           start_index,
    int           G,
    int           N,
    int           C4,
    int           M,
    float*        __restrict__ out,
    int64_t*      __restrict__ rev,
    float*        __restrict__ weight)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;

    const int batch_base = offsets[b];
    const int dst        = batch_base + start_index + group_prefix[b * M + m];
    const bool is_merged = mask[b * M + m];

    const int src_start = start_index + m * G;
    const int bN        = b * N;

    const float4* tokens_v = reinterpret_cast<const float4*>(tokens);
    float4*       out_v    = reinterpret_cast<float4*>(out);

    if (is_merged) {
        const float inv_G = 1.0f / static_cast<float>(G);
        for (int c4 = threadIdx.x; c4 < C4; c4 += blockDim.x) {
            float4 sum = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll 8
            for (int g = 0; g < G; ++g) {
                float4 v = tokens_v[(bN + src_start + g) * C4 + c4];
                sum.x += v.x;  sum.y += v.y;
                sum.z += v.z;  sum.w += v.w;
            }
            sum.x *= inv_G;  sum.y *= inv_G;
            sum.z *= inv_G;  sum.w *= inv_G;
            out_v[dst * C4 + c4] = sum;
        }
        if (threadIdx.x == 0) {
            weight[dst] = static_cast<float>(G);
        }
        for (int g = threadIdx.x; g < G; g += blockDim.x) {
            rev[bN + src_start + g] = static_cast<int64_t>(dst);
        }
    } else {
        for (int g = 0; g < G; ++g) {
            for (int c4 = threadIdx.x; c4 < C4; c4 += blockDim.x) {
                out_v[(dst + g) * C4 + c4] = tokens_v[(bN + src_start + g) * C4 + c4];
            }
        }
        for (int g = threadIdx.x; g < G; g += blockDim.x) {
            weight[dst + g] = 1.0f;
            rev[bN + src_start + g] = static_cast<int64_t>(dst + g);
        }
    }
}


// ==== Scalar fallback scatter kernel (int64 indexing for dst*C overflow safety) ====
template <typename scalar_t>
__global__ void token_merge_scatter_kernel(
    const scalar_t* __restrict__ tokens,
    const bool*     __restrict__ mask,
    const int*      __restrict__ group_prefix,
    const int*      __restrict__ offsets,
    int             start_index,
    int             G,
    int             N,
    int64_t         C,
    int             M,
    scalar_t*       __restrict__ out,
    int64_t*        __restrict__ rev,
    float*          __restrict__ weight)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;

    const int  batch_base = offsets[b];
    const int  dst        = batch_base + start_index + group_prefix[b * M + m];
    const bool is_merged  = mask[b * M + m];
    const int  src_start  = start_index + m * G;
    const int  bN         = b * N;

    if (is_merged) {
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            scalar_t sum = scalar_t(0);
            for (int g = 0; g < G; ++g) {
                sum += tokens[static_cast<int64_t>(bN + src_start + g) * C + c];
            }
            out[static_cast<int64_t>(dst) * C + c] = sum / static_cast<scalar_t>(G);
        }
        if (threadIdx.x == 0) {
            weight[dst] = static_cast<float>(G);
        }
        for (int g = threadIdx.x; g < G; g += blockDim.x) {
            rev[bN + src_start + g] = static_cast<int64_t>(dst);
        }
    } else {
        for (int g = 0; g < G; ++g) {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                out[static_cast<int64_t>(dst + g) * C + c] =
                    tokens[static_cast<int64_t>(bN + src_start + g) * C + c];
            }
        }
        for (int g = threadIdx.x; g < G; g += blockDim.x) {
            weight[dst + g] = 1.0f;
            rev[bN + src_start + g] = static_cast<int64_t>(dst + g);
        }
    }
}


// ==== Special-token kernel ====
template <typename scalar_t>
__global__ void token_merge_special_kernel(
    const scalar_t* __restrict__ tokens,
    const int*      __restrict__ offsets,
    int             start_index,
    int             N,
    int64_t         C,
    scalar_t*       __restrict__ out,
    int64_t*        __restrict__ rev,
    float*          __restrict__ weight)
{
    const int b = blockIdx.x;
    const int batch_base = offsets[b];
    const int bN = b * N;

    for (int s = 0; s < start_index; ++s) {
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            out[static_cast<int64_t>(batch_base + s) * C + c] =
                tokens[static_cast<int64_t>(bN + s) * C + c];
        }
    }
    for (int s = threadIdx.x; s < start_index; s += blockDim.x) {
        weight[batch_base + s] = 1.0f;
        rev[bN + s] = static_cast<int64_t>(batch_base + s);
    }
}


// ==== Gather kernel for token_split (float4 vectorised) ====
__global__ void token_split_gather_kernel_vec4(
    const float*    __restrict__ merged,
    const int64_t*  __restrict__ rev_id,
    int64_t         C,
    int64_t         total,
    float*          __restrict__ out)
{
    const int64_t i = blockIdx.x;
    if (i >= total) return;

    const int64_t src = rev_id[i];
    const int64_t C4  = C >> 2;

    const float4* merged_v = reinterpret_cast<const float4*>(merged);
    float4*       out_v    = reinterpret_cast<float4*>(out);

    for (int c4 = threadIdx.x; c4 < C4; c4 += blockDim.x) {
        out_v[i * C4 + c4] = merged_v[src * C4 + c4];
    }
}


// ==== Gather kernel for token_split (scalar fallback) ====
template <typename scalar_t>
__global__ void token_split_gather_kernel(
    const scalar_t* __restrict__ merged,
    const int64_t*  __restrict__ rev_id,
    int64_t         C,
    int64_t         total,
    scalar_t*       __restrict__ out)
{
    const int64_t i = blockIdx.x;
    if (i >= total) return;

    const int64_t src = rev_id[i];
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        out[i * C + c] = merged[src * C + c];
    }
}


// =========================================================================
// Host entry: token_merge_cuda
// =========================================================================
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
co_me_token_merge_cuda(
    at::Tensor  tokens,
    int64_t     start_index,
    int64_t     group_size,
    at::Tensor  merge_mask)
{
    TORCH_CHECK(tokens.is_cuda(),      "tokens must be on CUDA");
    TORCH_CHECK(merge_mask.is_cuda(),  "merge_mask must be on CUDA");
    TORCH_CHECK(tokens.dim() == 3,     "tokens must be [B, N, C]");
    TORCH_CHECK(merge_mask.dim() == 2, "merge_mask must be [B, M]");

    auto tokens_c = tokens.contiguous();
    auto mask_c   = merge_mask.to(at::kBool).contiguous();

    const int64_t B = tokens_c.size(0);
    const int64_t N = tokens_c.size(1);
    const int64_t C = tokens_c.size(2);
    const int64_t M = mask_c.size(1);

    TORCH_CHECK(mask_c.size(0) == B, "merge_mask batch != tokens batch");
    TORCH_CHECK(N == start_index + M * group_size,
                "N != start_index + M * group_size");

    // 1. Compute total output size P from mask (no pipeline-stalling sync)
    //    P = B*N - total_merged*(G-1) where total_merged = number of True entries
    const int64_t total_merged = mask_c.sum().item<int64_t>();
    const int64_t P = B * N - total_merged * (group_size - 1);

    auto out = at::empty({P, C}, tokens_c.options());
    auto rev = at::empty({B * N}, tokens_c.options().dtype(at::kLong));
    auto weight = at::empty({P}, tokens_c.options().dtype(at::kFloat));

    // 2. BlockScan prefix sum + batch lengths in a single kernel (1 launch for all B rows)
    auto group_prefix = at::empty({B, M}, mask_c.options().dtype(at::kInt));
    auto L_b = at::empty({B}, mask_c.options().dtype(at::kInt));
    block_prefix_scan_kernel<<<B, SCAN_BLOCK>>>(
        mask_c.data_ptr<bool>(),
        static_cast<int>(group_size),
        static_cast<int>(M),
        static_cast<int>(start_index),
        group_prefix.data_ptr<int>(),
        L_b.data_ptr<int>());

    // 3. Prefix sum of L_b → offsets [B+1] (entirely on GPU)
    auto offsets = at::empty({B + 1}, tokens_c.options().dtype(at::kInt));
    offsets[0] = 0;
    offsets.narrow(0, 1, B).copy_(at::cumsum(L_b, 0));

    // 5. Launch scatter kernel for image-token groups
    constexpr int THREADS = 256;
    const bool use_vec4 = (tokens_c.scalar_type() == at::kFloat) && (C % 4 == 0);

    const int N_int = static_cast<int>(N);
    const int C4    = static_cast<int>(C >> 2);
    const int M_int = static_cast<int>(M);
    const int S_int = static_cast<int>(start_index);
    const int G_int = static_cast<int>(group_size);

    if (M > 0) {
        dim3 grid(B, M);
        if (use_vec4) {
            token_merge_scatter_kernel_vec4<<<grid, THREADS>>>(
                tokens_c.data_ptr<float>(),
                mask_c.data_ptr<bool>(),
                group_prefix.data_ptr<int>(),
                offsets.data_ptr<int>(),
                S_int, G_int, N_int, C4, M_int,
                out.data_ptr<float>(),
                rev.data_ptr<int64_t>(),
                weight.data_ptr<float>());
        } else {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16,
                tokens_c.scalar_type(), "token_merge_scatter", ([&] {
                    token_merge_scatter_kernel<scalar_t><<<grid, THREADS>>>(
                        tokens_c.data_ptr<scalar_t>(),
                        mask_c.data_ptr<bool>(),
                        group_prefix.data_ptr<int>(),
                        offsets.data_ptr<int>(),
                        S_int, G_int, N_int, C, M_int,
                        out.data_ptr<scalar_t>(),
                        rev.data_ptr<int64_t>(),
                        weight.data_ptr<float>());
                }));
        }
    }

    // 6. Launch special-token kernel
    if (start_index > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16,
            tokens_c.scalar_type(), "token_merge_special", ([&] {
                token_merge_special_kernel<scalar_t><<<B, THREADS>>>(
                    tokens_c.data_ptr<scalar_t>(),
                    offsets.data_ptr<int>(),
                    S_int, N_int, C,
                    out.data_ptr<scalar_t>(),
                    rev.data_ptr<int64_t>(),
                    weight.data_ptr<float>());
            }));
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "token_merge_cuda: kernel launch failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(out, offsets, rev, weight);
}


// =========================================================================
// Host entry: token_split_cuda
// =========================================================================
at::Tensor
co_me_token_split_cuda(
    at::Tensor  merged_flat,
    at::Tensor  rev_id,
    int64_t     C)
{
    TORCH_CHECK(merged_flat.is_cuda(),  "merged_flat must be on CUDA");
    TORCH_CHECK(rev_id.is_cuda(),       "rev_id must be on CUDA");
    TORCH_CHECK(merged_flat.dim() == 2, "merged_flat must be [P, C]");
    TORCH_CHECK(merged_flat.size(1) == C, "C mismatch");

    auto merged_c = merged_flat.contiguous();
    auto rev_c    = rev_id.contiguous();
    const int64_t total = rev_c.size(0);

    auto out = at::empty({total, C}, merged_c.options());

    constexpr int THREADS = 256;
    const bool use_vec4 = (merged_c.scalar_type() == at::kFloat) && (C % 4 == 0);

    if (use_vec4) {
        token_split_gather_kernel_vec4<<<total, THREADS>>>(
            merged_c.data_ptr<float>(),
            rev_c.data_ptr<int64_t>(),
            C, total,
            out.data_ptr<float>());
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16,
            merged_c.scalar_type(), "token_split_gather", ([&] {
                token_split_gather_kernel<scalar_t><<<total, THREADS>>>(
                    merged_c.data_ptr<scalar_t>(),
                    rev_c.data_ptr<int64_t>(),
                    C, total,
                    out.data_ptr<scalar_t>());
            }));
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "token_split_cuda: kernel launch failed: ",
                cudaGetErrorString(err));

    return out;
}
