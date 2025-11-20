#include <torch/extension.h>
#include "kernel_tokenmerge2.h"


// Python bindings with detailed type information for stub generation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for Co-Me project";

    m.def("token_grpcontract_v2_merge", &token_grpcontract_v2_merge,
          "Contract and merge groups of tokens with optional averaging.\n\n"
          "Args:\n"
          "    prefix_cnt (int): number of prefix token slots to reserve.\n"
          "    tokens (torch.Tensor): [B, N, G, C] typed tensor (float32, float16, bfloat16, or int64).\n"
          "    dst_lidx (torch.Tensor): [B, N * G] int64 tensor of destination indices.\n"
          "    out_ptr (torch.Tensor | None): [B, R, C] shaped tensor, if provided will write result into this pointer.\n\n"
          "Returns:\n"
          "    torch.Tensor: [B, prefix_cnt + M, C] output tensor after contraction.\n");    
    
    m.def(
        "token_grpcontract_v2_scan",
        &token_grpcontract_v2_scan,
        "Scan and group-contract a boolean mask per batch.\n\n"
        "Args:\n"
        "    prefix_cnt (int64_t): number of prefix token slots to reserve.\n"
        "    G (int64_t): group size for repeat and contraction.\n"
        "    mask (torch::Tensor): [B, N] boolean (or uint8) tensor on CUDA.\n\n"
        "Returns:\n"
        "    torch::Tensor: [B, N * G] int64 tensor of destination indices "
        "(with prefix offset applied).\n"
    );
}
