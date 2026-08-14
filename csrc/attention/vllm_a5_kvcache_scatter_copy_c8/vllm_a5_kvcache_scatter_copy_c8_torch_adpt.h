/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#ifndef VLLM_A5_KVCACHE_SCATTER_COPY_C8_TORCH_ADPT_H
#define VLLM_A5_KVCACHE_SCATTER_COPY_C8_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_dsa_a5_kvcache_scatter_copy_c8_out(
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts)
{
    constexpr int64_t block_size = 128;
    constexpr int64_t packed_row_bytes = 656;
    constexpr int64_t copy_capacity = 16384;
    TORCH_CHECK(copy_counts.dim() == 1,
                "DSA A5 packed KSC copy_counts must be one-dimensional.");
    const int64_t batch = copy_counts.size(0);

    TORCH_CHECK(hbm_kv_bytes.dim() == 4 &&
                    hbm_kv_bytes.size(0) > 0 &&
                    hbm_kv_bytes.size(1) == block_size &&
                    hbm_kv_bytes.size(2) == 1 &&
                    hbm_kv_bytes.size(3) == packed_row_bytes &&
                    dram_kv_bytes.dim() == 4 &&
                    dram_kv_bytes.size(0) > 0 &&
                    dram_kv_bytes.size(1) == block_size &&
                    dram_kv_bytes.size(2) == 1 &&
                    dram_kv_bytes.size(3) == packed_row_bytes,
                "DSA A5 packed C8 caches must be [blocks,128,1,656].");
    TORCH_CHECK(hbm_kv_bytes.scalar_type() == at::kChar &&
                    dram_kv_bytes.scalar_type() == at::kChar,
                "DSA A5 packed C8 cache adapters require int8 byte views.");
    TORCH_CHECK(batch > 0 &&
                    hbm_block_table.dim() == 2 &&
                    hbm_block_table.size(0) == batch &&
                    dram_block_table.dim() == 2 &&
                    dram_block_table.size(0) == batch &&
                    source_token_ids.dim() == 3 &&
                    source_token_ids.size(0) == batch &&
                    source_token_ids.size(1) == 1 &&
                    source_token_ids.size(2) == copy_capacity &&
                    destination_slots.sizes() == source_token_ids.sizes(),
                "DSA A5 packed KSC metadata shapes are inconsistent.");
    const auto device = hbm_kv_bytes.device();
    const at::Tensor* tensors[] = {
        &hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table,
        &dram_block_table, &source_token_ids, &destination_slots,
        &copy_counts};
    for (const at::Tensor* tensor : tensors) {
        TORCH_CHECK(tensor->device() == device,
                    "all DSA A5 packed KSC tensors must share one device.");
        TORCH_CHECK(tensor->is_contiguous(),
                    "all DSA A5 packed KSC tensors must be contiguous.");
    }
    const at::Tensor* metadata_tensors[] = {
        &hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts};
    for (const at::Tensor* tensor : metadata_tensors) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "DSA A5 packed KSC metadata must be int32.");
    }

    // OpDef declares hbm_kv as an in-place reference, so the generated ACLNN
    // workspace API accepts the mutable tensor once rather than as two args.
    EXEC_NPU_CMD(aclnnVllmA5KvcacheScatterCopyC8,
                 hbm_kv_bytes,
                 dram_kv_bytes,
                 hbm_block_table,
                 dram_block_table,
                 source_token_ids,
                 destination_slots,
                 copy_counts);
}

}  // namespace vllm_ascend

#endif  // VLLM_A5_KVCACHE_SCATTER_COPY_C8_TORCH_ADPT_H
