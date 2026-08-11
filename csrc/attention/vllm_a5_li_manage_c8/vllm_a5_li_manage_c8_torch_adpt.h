/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#ifndef VLLM_A5_LI_MANAGE_C8_TORCH_ADPT_H
#define VLLM_A5_LI_MANAGE_C8_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_dsa_a5_li_manage_c8_out(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& row_modes,
    const at::Tensor& actual_seq_lengths_key,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts,
    at::Tensor tail_info)
{
    constexpr int64_t sparse_count = 2048;
    constexpr int64_t output_capacity = 16384;
    constexpr int64_t max_token_capacity = 1 << 18;
    const int64_t batch = topk_indices.size(0);

    TORCH_CHECK(topk_indices.dim() == 3 && batch > 0 &&
                    topk_indices.size(1) == 1 &&
                    topk_indices.size(2) == sparse_count,
                "DSA A5 LI manager topk_indices must be [B,1,2048].");
    TORCH_CHECK(req_pool_entries.dim() == 1 &&
                    req_pool_entries.size(0) == batch &&
                    row_modes.dim() == 1 && row_modes.size(0) == batch &&
                    actual_seq_lengths_key.dim() == 1 &&
                    actual_seq_lengths_key.size(0) == batch,
                "DSA A5 LI manager row metadata shapes are inconsistent.");
    TORCH_CHECK(cache_slots_pool.dim() == 2 &&
                    cache_slots_pool.size(0) > 0 &&
                    cache_slots_pool.size(1) >= 2 &&
                    cache_slots_pool.size(1) <= max_token_capacity + 1,
                "DSA A5 LI manager cache_slots_pool shape is invalid.");
    TORCH_CHECK(source_ids.dim() == 3 &&
                    source_ids.size(0) == batch &&
                    source_ids.size(1) == 1 &&
                    source_ids.size(2) == output_capacity &&
                    destination_slots.sizes() == source_ids.sizes() &&
                    miss_counts.dim() == 1 &&
                    miss_counts.size(0) == batch &&
                    tail_info.dim() == 2 &&
                    tail_info.size(0) == batch &&
                    tail_info.size(1) == 2,
                "DSA A5 LI manager output shapes are invalid.");

    const auto device = topk_indices.device();
    const at::Tensor* tensors[] = {
        &topk_indices, &req_pool_entries, &cache_slots_pool,
        &row_modes, &actual_seq_lengths_key, &source_ids,
        &destination_slots, &miss_counts, &tail_info};
    for (const at::Tensor* tensor : tensors) {
        TORCH_CHECK(tensor->device() == device,
                    "all DSA A5 LI manager tensors must share one device.");
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "all DSA A5 LI manager tensors must be int32.");
        TORCH_CHECK(tensor->is_contiguous(),
                    "all DSA A5 LI manager tensors must be contiguous.");
    }

    EXEC_NPU_CMD(aclnnVllmA5LiManageC8,
                 topk_indices,
                 req_pool_entries,
                 cache_slots_pool,
                 row_modes,
                 actual_seq_lengths_key,
                 source_ids,
                 destination_slots,
                 miss_counts,
                 tail_info);
}

}  // namespace vllm_ascend

#endif  // VLLM_A5_LI_MANAGE_C8_TORCH_ADPT_H
