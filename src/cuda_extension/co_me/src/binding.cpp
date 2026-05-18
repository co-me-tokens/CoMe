#include <torch/extension.h>
#include "kernel_token_merge.h"
#include "kernel_token_merge_legacy.h"


// Python bindings with detailed type information for stub generation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for Co-Me project";

    m.def("co_me_token_merge_cuda_legacy", &co_me_token_merge_cuda_legacy,
          "Contract and merge groups of tokens with optional averaging.\n\n"
          "Args:\n"
          "    prefix_cnt (int): number of prefix token slots to reserve.\n"
          "    tokens (torch.Tensor): [B, N, G, C] typed tensor (float32, float16, bfloat16, or int64).\n"
          "    dst_lidx (torch.Tensor): [B, N * G] int64 tensor of destination indices.\n"
          "    out_ptr (torch.Tensor | None): [B, R, C] shaped tensor, if provided will write result into this pointer.\n\n"
          "Returns:\n"
          "    torch.Tensor: [B, prefix_cnt + M, C] output tensor after contraction.\n");    
    
    m.def(
        "co_me_token_merge_scan_legacy",
        &co_me_token_merge_scan_legacy,
        "Scan and group-contract a boolean mask per batch.\n\n"
        "Args:\n"
        "    prefix_cnt (int64_t): number of prefix token slots to reserve.\n"
        "    G (int64_t): group size for repeat and contraction.\n"
        "    mask (torch::Tensor): [B, N] boolean (or uint8) tensor on CUDA.\n\n"
        "Returns:\n"
        "    torch::Tensor: [B, N * G] int64 tensor of destination indices "
        "(with prefix offset applied).\n"
    );

    m.def("co_me_token_merge_cuda", &co_me_token_merge_cuda,
          "Fused token merge: scan + scatter + rev_id.\n\n"
          "Args:\n"
          "    tokens (torch.Tensor): [B, N, C] float tensor on CUDA.\n"
          "    start_index (int): number of special prefix tokens per sample.\n"
          "    group_size (int): G, tokens per merge group.\n"
          "    merge_mask (torch.Tensor): [B, M] bool tensor on CUDA.\n\n"
          "Returns:\n"
          "    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n"
          "        out_flat [P, C], offsets [B+1], rev_id [B*N], weight [P].\n");

    m.def("co_me_token_split_cuda", &co_me_token_split_cuda,
          "Token split (inverse of merge): gather by rev_id.\n\n"
          "Args:\n"
          "    merged_flat (torch.Tensor): [P, C] float tensor on CUDA.\n"
          "    rev_id (torch.Tensor): [total] int64 gather indices.\n"
          "    C (int): feature dimension.\n\n"
          "Returns:\n"
          "    torch.Tensor: [total, C] restored tokens.\n");
}
