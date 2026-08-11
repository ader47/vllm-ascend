#ifndef KV_CACHE_FULL_BLOCK_DUMP_C8_TORCH_ADPT_H
#define KV_CACHE_FULL_BLOCK_DUMP_C8_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_kv_cache_full_block_dump_c8(
    const at::Tensor& src_cache,
    const at::Tensor& dst_cache,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids)
{
    TORCH_CHECK(src_cache.device().is_privateuseone() &&
                    dst_cache.device().is_privateuseone(),
                "packed C8 dump caches must be on NPU");
    TORCH_CHECK(src_cache.device() == dst_cache.device() &&
                    src_block_ids.device() == src_cache.device() &&
                    dst_block_ids.device() == src_cache.device(),
                "packed C8 dump tensors must share one NPU device");
    TORCH_CHECK(src_cache.scalar_type() == at::ScalarType::Char &&
                    dst_cache.scalar_type() == at::ScalarType::Char,
                "packed C8 dump cache byte views must be int8");
    TORCH_CHECK(src_cache.dim() == 4 && dst_cache.dim() == 4 &&
                    src_cache.size(1) == 128 && dst_cache.size(1) == 128 &&
                    src_cache.size(2) == 1 && dst_cache.size(2) == 1 &&
                    src_cache.size(3) == 656 && dst_cache.size(3) == 656,
                "packed C8 dump caches must be [blocks, 128, 1, 656]");
    TORCH_CHECK(src_cache.is_contiguous() && dst_cache.is_contiguous(),
                "packed C8 dump caches must be contiguous");
    TORCH_CHECK(src_block_ids.dim() == 1 && dst_block_ids.dim() == 1 &&
                    src_block_ids.numel() == dst_block_ids.numel(),
                "packed C8 dump block-id tensors must have matching rows");
    TORCH_CHECK(src_block_ids.scalar_type() == at::ScalarType::Int &&
                    dst_block_ids.scalar_type() == at::ScalarType::Int &&
                    src_block_ids.is_contiguous() &&
                    dst_block_ids.is_contiguous(),
                "packed C8 dump block ids must be contiguous int32 tensors");
    TORCH_CHECK(src_cache.data_ptr() != dst_cache.data_ptr(),
                "packed C8 dump source and destination arenas must not alias");

    EXEC_NPU_CMD(aclnnKvCacheFullBlockDumpC8,
                 src_cache,
                 dst_cache,
                 src_block_ids,
                 dst_block_ids);
}

}  // namespace vllm_ascend

#endif
