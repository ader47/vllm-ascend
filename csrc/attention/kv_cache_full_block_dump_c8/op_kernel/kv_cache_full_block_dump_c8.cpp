/*
 * Copy complete packed C8 KV-cache blocks from HBM to swapped DRAM.
 * The payload is opaque: every byte, including RoPE and quantization scales,
 * is moved without interpretation.
 */
#include "kernel_operator.h"
#include "kv_cache_full_block_dump_c8_tiling.h"

namespace {
using namespace AscendC;

constexpr int32_t NOOP_DST_BLOCK_ID = -1;

class KvCacheFullBlockDumpC8Kernel {
public:
    __aicore__ inline KvCacheFullBlockDumpC8Kernel(
        TPipe *pipe,
        const KvCacheFullBlockDumpC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR srcCache,
        GM_ADDR dstCache,
        GM_ADDR srcBlockIds,
        GM_ADDR dstBlockIds)
    {
        coreIdx_ = GetBlockIdx();
        dstBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t *)dstBlockIds);
        if (!HasAssignedDump()) {
            return;
        }

        srcCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)srcCache);
        dstCacheGm_.SetGlobalBuffer((__gm__ uint8_t *)dstCache);
        srcBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t *)srcBlockIds);
        const uint32_t alignedChunkBytes =
            (tiling_->chunkBytes + 31U) & ~31U;
        pipe_->InitBuffer(copyQueue_, 2, alignedChunkBytes);
        active_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }

        for (uint64_t task = coreIdx_; task < tiling_->taskCount;
             task += tiling_->usedCoreNum) {
            const uint32_t row = static_cast<uint32_t>(
                task / tiling_->tasksPerRow);
            const uint32_t chunk = static_cast<uint32_t>(
                task - static_cast<uint64_t>(row) * tiling_->tasksPerRow);
            const int32_t dstBlock = dstBlockIdsGm_.GetValue(row);
            if (dstBlock == NOOP_DST_BLOCK_ID) {
                continue;
            }
            ASSERT_MSG(dstBlock >= 0,
                       "packed C8 dump destination block id must be >= -1");
            const int32_t srcBlock = srcBlockIdsGm_.GetValue(row);
            ASSERT_MSG(srcBlock >= 0,
                       "packed C8 dump source block id must be non-negative");
            ASSERT_MSG(
                srcBlock < static_cast<int32_t>(tiling_->srcBlockNum),
                "packed C8 dump source block id exceeds capacity");
            ASSERT_MSG(
                dstBlock < static_cast<int32_t>(tiling_->dstBlockNum),
                "packed C8 dump destination block id exceeds capacity");

            const uint32_t byteOffset = chunk * tiling_->chunkBytes;
            const uint32_t remaining = tiling_->bytesPerBlock - byteOffset;
            const uint32_t copyBytes = remaining < tiling_->chunkBytes
                ? remaining : tiling_->chunkBytes;
            const uint64_t srcOffset =
                static_cast<uint64_t>(srcBlock) * tiling_->bytesPerBlock +
                byteOffset;
            const uint64_t dstOffset =
                static_cast<uint64_t>(dstBlock) * tiling_->bytesPerBlock +
                byteOffset;

            LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyExtParams params{1, copyBytes, 0, 0, 0};
            DataCopyPad<uint8_t, PaddingMode::Normal>(
                local, srcCacheGm_[srcOffset], params, pad);
            copyQueue_.EnQue<uint8_t>(local);
            local = copyQueue_.DeQue<uint8_t>();
            DataCopyPad<uint8_t, PaddingMode::Normal>(
                dstCacheGm_[dstOffset], local, params);
            copyQueue_.FreeTensor(local);
        }
    }

private:
    __aicore__ inline bool HasAssignedDump()
    {
        for (uint64_t task = coreIdx_; task < tiling_->taskCount;
             task += tiling_->usedCoreNum) {
            const uint32_t row = static_cast<uint32_t>(
                task / tiling_->tasksPerRow);
            const int32_t dstBlock = dstBlockIdsGm_.GetValue(row);
            ASSERT_MSG(dstBlock >= NOOP_DST_BLOCK_ID,
                       "packed C8 dump destination block id must be >= -1");
            if (dstBlock != NOOP_DST_BLOCK_ID) {
                return true;
            }
        }
        return false;
    }

    TPipe *pipe_;
    const KvCacheFullBlockDumpC8TilingData *tiling_;
    uint32_t coreIdx_ = 0;
    bool active_ = false;
    GlobalTensor<uint8_t> srcCacheGm_;
    GlobalTensor<uint8_t> dstCacheGm_;
    GlobalTensor<int32_t> srcBlockIdsGm_;
    GlobalTensor<int32_t> dstBlockIdsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};
}  // namespace

extern "C" __global__ __aicore__ void kv_cache_full_block_dump_c8(
    GM_ADDR srcCache,
    GM_ADDR dstCache,
    GM_ADDR srcBlockIds,
    GM_ADDR dstBlockIds,
    GM_ADDR dstCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)dstCacheOut;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KvCacheFullBlockDumpC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KvCacheFullBlockDumpC8Kernel op(&pipe, &tilingData);
    op.Init(srcCache, dstCache, srcBlockIds, dstBlockIds);
    op.Process();
}
