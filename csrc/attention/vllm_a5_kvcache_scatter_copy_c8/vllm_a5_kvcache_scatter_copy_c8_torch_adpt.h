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
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv,
    at::Tensor attention_slots,
    at::Tensor resident_seq_lengths)
{
    constexpr int64_t block_size = 128;
    constexpr int64_t packed_row_bytes = 656;
    constexpr int64_t copy_capacity = 16384;
    constexpr int64_t attention_capacity = 2176;
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
    TORCH_CHECK(batch > 0 && cache_tokens.dim() == 1 &&
                    cache_tokens.size(0) == batch &&
                    candidate_lens.dim() == 1 &&
                    candidate_lens.size(0) == batch &&
                    actual_seq_lengths_kv.dim() == 1 &&
                    actual_seq_lengths_kv.size(0) == batch &&
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
    TORCH_CHECK(attention_slots.dim() == 3 &&
                    attention_slots.size(0) == batch &&
                    attention_slots.size(1) == 1 &&
                    attention_slots.size(2) == attention_capacity &&
                    resident_seq_lengths.dim() == 1 &&
                    resident_seq_lengths.size(0) == batch,
                "DSA A5 packed KSC outputs must be [B,1,2176] and [B].");

    const auto device = hbm_kv_bytes.device();
    const at::Tensor* tensors[] = {
        &hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table,
        &dram_block_table, &source_token_ids, &destination_slots,
        &copy_counts, &cache_tokens, &candidate_lens,
        &actual_seq_lengths_kv, &attention_slots,
        &resident_seq_lengths};
    for (const at::Tensor* tensor : tensors) {
        TORCH_CHECK(tensor->device() == device,
                    "all DSA A5 packed KSC tensors must share one device.");
        TORCH_CHECK(tensor->is_contiguous(),
                    "all DSA A5 packed KSC tensors must be contiguous.");
    }
    const at::Tensor* metadata_tensors[] = {
        &hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts, &cache_tokens,
        &candidate_lens, &actual_seq_lengths_kv, &attention_slots,
        &resident_seq_lengths};
    for (const at::Tensor* tensor : metadata_tensors) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "DSA A5 packed KSC metadata must be int32.");
    }

    EXEC_NPU_CMD(aclnnVllmA5KvcacheScatterCopyC8,
                 hbm_kv_bytes,
                 dram_kv_bytes,
                 hbm_block_table,
                 dram_block_table,
                 source_token_ids,
                 destination_slots,
                 copy_counts,
                 cache_tokens,
                 candidate_lens,
                 actual_seq_lengths_kv,
                 attention_slots,
                 resident_seq_lengths);
}

}  // namespace vllm_ascend

#endif  // VLLM_A5_KVCACHE_SCATTER_COPY_C8_TORCH_ADPT_H
