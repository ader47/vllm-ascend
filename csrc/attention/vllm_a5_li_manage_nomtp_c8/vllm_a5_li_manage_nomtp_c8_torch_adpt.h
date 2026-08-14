/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#ifndef VLLM_A5_LI_MANAGE_NOMTP_C8_TORCH_ADPT_H
#define VLLM_A5_LI_MANAGE_NOMTP_C8_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_dsa_a5_li_manage_nomtp_c8_out(
    const at::Tensor& index_weights,
    const at::Tensor& query,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_block_table,
    const at::Tensor& candidate_lens,
    const at::Tensor& final_seq_lengths_kv,
    const at::Tensor& row_modes,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor sparse_and_tail_slots,
    at::Tensor resident_seq_lengths,
    at::Tensor copy_src_ids,
    at::Tensor copy_dst_slots,
    at::Tensor copy_counts)
{
    constexpr int64_t block_size = 128;
    constexpr int64_t head_dim = 128;
    constexpr int64_t attention_capacity = 2176;
    constexpr int64_t copy_capacity = 16384;
    constexpr int64_t max_source_capacity = 1 << 18;

    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 &&
                    (query.size(1) == 32 || query.size(1) == 64) &&
                    query.size(2) == head_dim &&
                    query.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "DSA A5 fused LIDU query must be float8_e4m3fn "
                "[B,32|64,128].");
    const int64_t batch = query.size(0);
    const int64_t heads = query.size(1);
    TORCH_CHECK(index_weights.dim() == 2 &&
                    index_weights.size(0) == batch &&
                    index_weights.size(1) == heads &&
                    index_weights.scalar_type() == at::kBFloat16 &&
                    index_weights.stride(1) == 1,
                "DSA A5 fused LIDU weights must be BF16 [B,N_IDX] "
                "with a contiguous head axis.");
    TORCH_CHECK(query_dequant_scale.dim() == 2 &&
                    query_dequant_scale.size(0) == batch &&
                    query_dequant_scale.size(1) == heads &&
                    query_dequant_scale.scalar_type() == at::kFloat,
                "DSA A5 fused LIDU query scales must be FP32 [B,N_IDX].");
    TORCH_CHECK(index_key_cache.dim() == 4 &&
                    index_key_cache.size(0) > 0 &&
                    index_key_cache.size(1) == block_size &&
                    index_key_cache.size(2) == 1 &&
                    index_key_cache.size(3) == head_dim &&
                    index_key_cache.scalar_type() ==
                        at::ScalarType::Float8_e4m3fn,
                "DSA A5 fused LIDU key cache must be float8_e4m3fn "
                "[blocks,128,1,128].");
    TORCH_CHECK(index_key_dequant_scale.dim() == 3 &&
                    index_key_dequant_scale.size(0) ==
                        index_key_cache.size(0) &&
                    index_key_dequant_scale.size(1) == block_size &&
                    index_key_dequant_scale.size(2) == 1 &&
                    index_key_dequant_scale.scalar_type() == at::kFloat,
                "DSA A5 fused LIDU key scales must be FP32 "
                "[blocks,128,1].");
    TORCH_CHECK(index_key_cache.stride(1) == head_dim &&
                    index_key_cache.stride(2) == head_dim &&
                    index_key_cache.stride(3) == 1 &&
                    index_key_dequant_scale.stride(1) == 1 &&
                    index_key_dequant_scale.stride(2) == 1,
                "DSA A5 fused LIDU permits padding only on the Indexer "
                "cache block axis.");

    const at::Tensor* int_tensors[] = {
        &actual_seq_lengths_query, &index_block_table, &candidate_lens,
        &final_seq_lengths_kv, &row_modes, &req_pool_entries,
        &cache_slots_pool, &sparse_and_tail_slots, &resident_seq_lengths,
        &copy_src_ids, &copy_dst_slots, &copy_counts};
    for (const at::Tensor* tensor : int_tensors) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "DSA A5 fused LIDU metadata and outputs must be int32.");
    }
    TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                    actual_seq_lengths_query.size(0) == batch &&
                    index_block_table.dim() == 2 &&
                    index_block_table.size(0) == batch &&
                    index_block_table.size(1) > 0 &&
                    index_block_table.size(1) * block_size <=
                        max_source_capacity &&
                    candidate_lens.dim() == 1 &&
                    candidate_lens.size(0) == batch &&
                    final_seq_lengths_kv.dim() == 1 &&
                    final_seq_lengths_kv.size(0) == batch &&
                    row_modes.dim() == 1 && row_modes.size(0) == batch &&
                    req_pool_entries.dim() == 1 &&
                    req_pool_entries.size(0) == batch &&
                    cache_slots_pool.dim() == 2 &&
                    cache_slots_pool.size(0) > 0 &&
                    cache_slots_pool.size(1) >= 2 &&
                    cache_slots_pool.size(1) <= max_source_capacity + 1,
                "DSA A5 fused LIDU request metadata shapes are invalid.");
    TORCH_CHECK(sparse_and_tail_slots.dim() == 3 &&
                    sparse_and_tail_slots.size(0) == batch &&
                    sparse_and_tail_slots.size(1) == 1 &&
                    sparse_and_tail_slots.size(2) == attention_capacity &&
                    resident_seq_lengths.dim() == 1 &&
                    resident_seq_lengths.size(0) == batch &&
                    copy_src_ids.dim() == 3 &&
                    copy_src_ids.size(0) == batch &&
                    copy_src_ids.size(1) == 1 &&
                    copy_src_ids.size(2) == copy_capacity &&
                    copy_dst_slots.sizes() == copy_src_ids.sizes() &&
                    copy_counts.dim() == 1 && copy_counts.size(0) == batch,
                "DSA A5 fused LIDU caller-owned output shapes are invalid.");

    const auto device = query.device();
    const at::Tensor* tensors[] = {
        &index_weights, &query, &query_dequant_scale,
        &actual_seq_lengths_query, &index_key_cache,
        &index_key_dequant_scale, &index_block_table, &candidate_lens,
        &final_seq_lengths_kv, &row_modes, &req_pool_entries,
        &cache_slots_pool, &sparse_and_tail_slots, &resident_seq_lengths,
        &copy_src_ids, &copy_dst_slots, &copy_counts};
    for (const at::Tensor* tensor : tensors) {
        TORCH_CHECK(tensor->device() == device,
                    "all DSA A5 fused LIDU tensors must share one device.");
    }
    const at::Tensor* contiguous_tensors[] = {
        &query, &query_dequant_scale, &actual_seq_lengths_query,
        &index_block_table, &candidate_lens, &final_seq_lengths_kv,
        &row_modes, &req_pool_entries, &cache_slots_pool,
        &sparse_and_tail_slots, &resident_seq_lengths, &copy_src_ids,
        &copy_dst_slots, &copy_counts};
    for (const at::Tensor* tensor : contiguous_tensors) {
        TORCH_CHECK(tensor->is_contiguous(),
                    "DSA A5 fused LIDU metadata and outputs must be contiguous.");
    }

    // EXEC_NPU_CMD forwards arguments through ConvertTypes(Ts&...), so
    // tensor.stride(0) temporaries cannot be passed directly. Keep the
    // element strides as named lvalues for the generated ACLNN attributes.
    int64_t key_stride = index_key_cache.stride(0);
    int64_t scale_stride = index_key_dequant_scale.stride(0);
    int64_t weight_stride = index_weights.stride(0);
    EXEC_NPU_CMD(aclnnVllmA5LiManageNomtpC8,
                 index_weights,
                 query,
                 query_dequant_scale,
                 actual_seq_lengths_query,
                 index_key_cache,
                 index_key_dequant_scale,
                 index_block_table,
                 candidate_lens,
                 final_seq_lengths_kv,
                 row_modes,
                 req_pool_entries,
                 cache_slots_pool,
                 key_stride,
                 scale_stride,
                 weight_stride,
                 sparse_and_tail_slots,
                 resident_seq_lengths,
                 copy_src_ids,
                 copy_dst_slots,
                 copy_counts);
}

}  // namespace vllm_ascend

#endif  // VLLM_A5_LI_MANAGE_NOMTP_C8_TORCH_ADPT_H
