#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <vector>

#include "src/namespace_config.h"

namespace FLASH_NAMESPACE {

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q,
    at::Tensor const& k,
    at::Tensor const& v,
    std::optional<at::Tensor>& out_,
    at::Tensor const& cu_seqlens_q,
    at::Tensor const& cu_seqlens_k,
    std::optional<at::Tensor>& seqused_k,
    std::optional<at::Tensor const>& leftpad_k_,
    std::optional<at::Tensor>& block_table_,
    std::optional<at::Tensor>& alibi_slopes_,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool return_softmax,
    std::optional<at::Generator> gen_
);

std::vector<at::Tensor> mha_varlen_fwd_w_perkey_bias(
    at::Tensor& q,
    at::Tensor const& k,
    at::Tensor const& v,
    std::optional<at::Tensor>& out_,
    at::Tensor const& cu_seqlens_q,
    at::Tensor const& cu_seqlens_k,
    std::optional<at::Tensor>& seqused_k,
    std::optional<at::Tensor const>& leftpad_k_,
    std::optional<at::Tensor>& block_table_,
    std::optional<at::Tensor>& alibi_slopes_,
    std::optional<at::Tensor>& perkey_bias_,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float softcap,
    bool return_softmax,
    std::optional<at::Generator> gen_
);

}  // namespace FLASH_NAMESPACE
