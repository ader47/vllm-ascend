#ifndef KV_CACHE_FULL_BLOCK_DUMP_C8_TILING_H
#define KV_CACHE_FULL_BLOCK_DUMP_C8_TILING_H

#include <cstdint>

struct KvCacheFullBlockDumpC8TilingData {
    uint32_t usedCoreNum;
    uint32_t rowCount;
    uint32_t srcBlockNum;
    uint32_t dstBlockNum;
    uint32_t bytesPerBlock;
    uint32_t chunkBytes;
    uint32_t tasksPerRow;
    uint64_t taskCount;
};

#endif
