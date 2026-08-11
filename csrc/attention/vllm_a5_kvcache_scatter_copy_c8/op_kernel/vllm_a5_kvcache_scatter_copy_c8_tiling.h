#ifndef VLLM_A5_KVCACHE_SCATTER_COPY_C8_TILING_H
#define VLLM_A5_KVCACHE_SCATTER_COPY_C8_TILING_H

#include <cstdint>

struct VllmA5KvcacheScatterCopyC8TilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t copyCap;
    uint32_t hbmMaxBlockNum;
    uint32_t dramMaxBlockNum;
    uint32_t hbmPhysicalBlockCount;
    uint32_t dramPhysicalBlockCount;
    uint32_t packedRowBytes;
    uint32_t attentionCapacity;
    uint64_t totalPairSlots;
};

#endif
