#include <kernel_tokenmerge2.h>
#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h> 
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/util/Optional.h>
#include <ATen/MemoryOverlap.h>
#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h> 
#include <vector>


template <typename scalar_t>
__global__ void token_reduce_kernel_v2(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> x,
    const at::PackedTensorAccessor64<int64_t, 2, at::RestrictPtrTraits> idx,
    at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> out,
    const int  G,
    const int  C)
{
    const int  b_id = blockIdx.x;          // 0 … B-1
    const int  g_id = blockIdx.y;          // 0 … N-1
    const int  row0 = g_id * G;            // first row in this group

    // Compute once per block; broadcast via shared memory
    __shared__ int64_t s_i0;
    __shared__ int     s_caseA; // 1 if Case A, 0 otherwise

    if (threadIdx.x == 0) {
        const int64_t i0 = idx[b_id][row0];
        // Add bounds check for safety
        const int64_t i1 = (row0 + 1 < idx.size(1)) ? idx[b_id][row0 + 1] : i0 + G;
        s_i0    = i0;
        s_caseA = (i1 - i0 == 1) ? 1 : 0;
    }
    __syncthreads();

    // Tell the compiler/optimizer this branch is uniform per warp
    const int branchA = __all_sync(0xFFFFFFFFu, s_caseA);

    if (branchA)  // —— Case A: straight copies ——
    {
        for (int c = threadIdx.x; c < C; c += blockDim.x)
        {
            #pragma unroll
            for (int j = 0; j < G; ++j)
            {
                const int  src_row    = row0 + j;
                const int64_t dst_row = s_i0 + j;
                out[b_id][dst_row][c] = x[b_id][src_row][c];
            }
        }
    }
    else          // —— Case B: mean across G and write once ——
    {
        for (int c = threadIdx.x; c < C; c += blockDim.x)
        {
            scalar_t sum = scalar_t(0);
            #pragma unroll
            for (int j = 0; j < G; ++j)
            {
                sum += x[b_id][row0 + j][c];
            }
            out[b_id][s_i0][c] = sum / static_cast<scalar_t>(G);
        }
    }
}


void token_reduce_cuda_v2(
        at::Tensor x,    // B×(N·G)×C, contiguous
        at::Tensor idx,  // B×(N·G)
        at::Tensor out,  // B×N×C, contiguous
        const int G)
{
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && out.is_cuda(), "All tensors must reside on the same CUDA device");
    TORCH_CHECK(x.size(1) % G == 0, "x.size(1) must be divisible by G");

    const int B = x.size(0);
    const int N = x.size(1) / G;
    const int C = x.size(2);

    constexpr int THREADS = 256;                 // good default for C ≥ 1 k
    const dim3  block(THREADS);
    const dim3  grid(B, N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "token_reduce_kernel_v2", ([&] {
            token_reduce_kernel_v2<scalar_t><<<grid, block>>>(
                x.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                idx.packed_accessor64<int64_t, 2, at::RestrictPtrTraits>(),
                out.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>(),
                G, C);
        }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "token_reduce_kernel_v2: kernel launch failed: ", cudaGetErrorString(err));
}


at::Tensor token_grpcontract_v2_merge(const int64_t prefix_cnt, at::Tensor tokens, at::Tensor dst_lidx, c10::optional<at::Tensor> out_ptr) {
    TORCH_CHECK(tokens.is_cuda() && dst_lidx.is_cuda(), "tokens and dst_lidx must be on CUDA device.");
    TORCH_CHECK(tokens.dim() == 4, "token_grpcontract: tokens must have shape [B, N, G, C]");
    TORCH_CHECK(dst_lidx.dim() == 2 && dst_lidx.size(0) == tokens.size(0) && dst_lidx.size(1) == (tokens.size(1) * tokens.size(2)), "dst_lidx must have shape [B, N*G]");

    at::Tensor out;
    const int64_t B = tokens.size(0);
    const int64_t N = tokens.size(1);
    const int64_t G = tokens.size(2);
    const int64_t C = tokens.size(3);
    const int64_t rows = dst_lidx[0][dst_lidx.size(1) - 1].item<int64_t>() + 1;

    if (out_ptr.has_value()){
        out = *out_ptr;
        TORCH_CHECK(out.is_cuda(), "provided out_ptr must be a CUDA tensor.");
        TORCH_CHECK(out.scalar_type() == tokens.scalar_type(), "provided out_ptr must have same dtype as tokens.");
        TORCH_CHECK(out.dim() ==3  && out.size(0)==B && out.size(2)==C, "out must be [B, N, C] with same B and C as token.");
        TORCH_CHECK(out.is_non_overlapping_and_dense(), "To make life easier and the cost of kernel explicit, out must be non_overlapping and dense.");
        TORCH_CHECK(out.size(1) == rows, "Out must match the exact space requirement.");
        at::assert_no_overlap(tokens, out);              // Memory aliasing is a serious problem.
        at::assert_no_internal_overlap(out);        // In case of self-overlapping.
    } else {
        out = at::empty({B, rows, C}, tokens.options());           // [B, N*G + D, C]
    }

    auto flat_token = tokens.flatten(1, 2);
    token_reduce_cuda_v2(flat_token, dst_lidx, out, G);
    return out;
}


////////////////
namespace {
    template<typename index_t=int64_t>
    struct IncFunctor_v2 {
        const bool* __restrict__ mask_;
        index_t     G_;

        __host__ __device__ IncFunctor_v2(const bool* m, index_t G) : mask_(m), G_(G) {}

        __host__ __device__ index_t operator()(index_t flat_idx) const {
            // flat_idx goes from 0 to N*G-1
            // We want to check mask[flat_idx / G] for group membership
            index_t group_idx = flat_idx / G_;
            return mask_[group_idx] ? 1 : G_;
        }
    };
}


/* ------------------------------------------------------------------------- */
// Internal helper function for processing a single row
at::Tensor build_dst_lidx_cuda_v2_single_row(
    const at::Tensor& mask,   // [N] bool - single row
    int64_t              G,
    c10::Allocator*      allocator = c10::cuda::CUDACachingAllocator::get()) {

    using index_t = int64_t;
    const int64_t N   = mask.size(0);  // This is N, not B
    const int64_t NT  = N * G;         // Total elements for this row

    // 1. Allocate dst_lidx (length NT)
    at::Tensor dst_lidx = at::empty({NT}, mask.options().dtype(at::kLong));

    // 2. Build transform iterator that streams 'increment' values.
    const bool*  d_mask = mask.data_ptr<bool>();
    auto         counting  = thrust::counting_iterator<index_t>(0);
    auto         inc_iter  = thrust::make_transform_iterator(counting, IncFunctor_v2<index_t>(d_mask, G));

    // 3. CUB exclusive scan
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    
    // First call to get temp storage size
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, inc_iter, dst_lidx.data_ptr<index_t>(), NT);

    // Allocate temp storage
    auto temp_ptr = allocator->allocate(temp_bytes);
    d_temp = temp_ptr.get();

    // Actual scan
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, inc_iter, dst_lidx.data_ptr<index_t>(), NT);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUB scan failed: ", cudaGetErrorString(err));
    
    return dst_lidx;
}


// Wrapper to maintain original interface but process row-by-row internally
at::Tensor build_dst_lidx_cuda_v2(
    const at::Tensor& mask,   // [B] or [N] bool
    int64_t              G,
    int64_t              prefix_empty_cnt,
    c10::Allocator*      allocator = c10::cuda::CUDACachingAllocator::get()) {
    
    // This function is called from token_grpcontract_v2_scan with a 1D tensor (single row)
    // So we just need to handle the 1D case
    TORCH_CHECK(mask.dim() == 1, "build_dst_lidx_cuda_v2 expects 1D mask");
    return build_dst_lidx_cuda_v2_single_row(mask, G, allocator);
}


at::Tensor token_grpcontract_v2_scan(
    int64_t          prefix_cnt,
    int64_t          G,
    at::Tensor    mask)
{
    TORCH_CHECK(mask.dim() == 2, "mask must be 2D");
    TORCH_CHECK(mask.is_cuda(),    "mask must be on CUDA");
    TORCH_CHECK(mask.scalar_type() == at::kBool || 
                mask.scalar_type() == at::kByte,
                "mask must be bool or uint8");

    auto mask_contig = mask.to(at::kBool).contiguous();
    const int64_t B = mask_contig.size(0);
    const int64_t N = mask_contig.size(1);

    // Prepare a vector to hold each row's result
    std::vector<at::Tensor> rows;
    rows.reserve(B);

    for (int64_t b = 0; b < B; ++b) {
        // 1) extract row b -> 1D mask of length N
        auto row_mask = mask_contig[b].contiguous();  // Ensure contiguous
        
        TORCH_CHECK(row_mask.dim() == 1 && row_mask.size(0) == N, 
                    "Extracted row must be 1D with size N");

        // 2) exclusive‐scan the "inc" values (1 if true, G if false),
        //    producing dst_fidx: a LongTensor of length N*G.
        auto dst_fidx = build_dst_lidx_cuda_v2(row_mask, G, /*unused*/ 0);

        // 3) convert to group‐indices by integer‐dividing by G
        auto dst_lidx = at::floor_divide(dst_fidx, G);

        // 4) apply global prefix offset (if any)
        if (prefix_cnt != 0) {
            dst_lidx = dst_lidx + prefix_cnt;
        }

        // 5) reshape to [1, N*G] and store
        rows.push_back(dst_lidx.view({1, -1}));
    }

    // concatenate back into [B, N*G]
    return at::cat(rows, /*dim=*/0);
}
