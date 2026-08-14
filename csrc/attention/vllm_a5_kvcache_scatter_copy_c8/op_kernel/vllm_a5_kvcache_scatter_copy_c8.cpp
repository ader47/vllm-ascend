/**
 * Ascend 950 token-granular copy for GLM-5.x C8 packed MLA cache rows.
 *
 * A row is opaque here.  In the current vLLM-Ascend A5 ABI it contains
 * 512 FP8 latent bytes, 64 BF16 RoPE elements (128 bytes), and four fp32
 * scales: 512 + 64 * 2 + 4 * 4 = 656 bytes. Copying the row byte-for-byte keeps
 * the quantization payload and its scales inseparable.
 */

#include "kernel_operator.h"
#include "vllm_a5_kvcache_scatter_copy_c8_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1;

class VllmA5KvcacheScatterCopyC8Kernel {
public:
    __aicore__ inline VllmA5KvcacheScatterCopyC8Kernel(
        TPipe *pipe,
        const VllmA5KvcacheScatterCopyC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKv,
        GM_ADDR dramKv,
        GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable,
        GM_ADDR copySrcIds,
        GM_ADDR copyDstSlots,
        GM_ADDR copyCounts)
    {
        coreIdx_ = GetBlockIdx();
        const uint32_t packedBufferBytes =
            (tiling_->packedRowBytes + 31U) & ~31U;
        pipe_->InitBuffer(copyQueue_, 2, packedBufferBytes);
        hbmKvGm_.SetGlobalBuffer((__gm__ uint8_t *)hbmKv);
        dramKvGm_.SetGlobalBuffer((__gm__ uint8_t *)dramKv);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)dramBlockTable);
        copySrcIdsGm_.SetGlobalBuffer((__gm__ int32_t *)copySrcIds);
        copyDstSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)copyDstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);
    }

    __aicore__ inline void Process()
    {
        cachedBatch_ = static_cast<uint32_t>(-1);
        cachedCount_ = 0;
        uint64_t current = FindNextValid(coreIdx_);
        CopyAddress currentAddress;
        while (current < tiling_->totalPairSlots &&
               !Resolve(current, currentAddress)) {
            current = FindNextValid(current + tiling_->usedCoreNum);
        }
        if (current >= tiling_->totalPairSlots) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            uint64_t next = FindNextValid(current + tiling_->usedCoreNum);
            CopyAddress nextAddress;
            while (next < tiling_->totalPairSlots &&
                   !Resolve(next, nextAddress)) {
                next = FindNextValid(next + tiling_->usedCoreNum);
            }
            const bool hasNext = next < tiling_->totalPairSlots;
            if (hasNext) {
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            current = next;
            currentAddress = nextAddress;
        }
    }

private:
    struct CopyAddress {
        uint64_t sourceByteOffset = 0;
        uint64_t destinationByteOffset = 0;
    };

    __aicore__ inline uint64_t FirstOwnedAtOrAfter(uint64_t start) const
    {
        if (start <= coreIdx_) {
            return coreIdx_;
        }
        const uint64_t distance = start - coreIdx_;
        const uint64_t steps =
            (distance + tiling_->usedCoreNum - 1) / tiling_->usedCoreNum;
        return coreIdx_ + steps * tiling_->usedCoreNum;
    }

    __aicore__ inline uint64_t FindNextValid(uint64_t flatPair)
    {
        while (flatPair < tiling_->totalPairSlots) {
            const uint32_t batch = static_cast<uint32_t>(
                flatPair / tiling_->copyCap);
            const uint32_t copyIndex = static_cast<uint32_t>(
                flatPair - static_cast<uint64_t>(batch) * tiling_->copyCap);
            if (batch != cachedBatch_) {
                cachedCount_ = copyCountsGm_.GetValue(batch);
                ASSERT_MSG(cachedCount_ >= 0,
                           "A5 DSA fused LIDU reported an invalid row");
                ASSERT_MSG(
                    cachedCount_ <= static_cast<int32_t>(tiling_->copyCap),
                    "A5 DSA fused LIDU copy count exceeds capacity");
                if (cachedCount_ < 0) {
                    cachedCount_ = 0;
                } else if (cachedCount_ > static_cast<int32_t>(tiling_->copyCap)) {
                    cachedCount_ = static_cast<int32_t>(tiling_->copyCap);
                }
                cachedBatch_ = batch;
            }
            if (copyIndex < static_cast<uint32_t>(cachedCount_)) {
                return flatPair;
            }
            flatPair = FirstOwnedAtOrAfter(
                (static_cast<uint64_t>(batch) + 1) * tiling_->copyCap);
        }
        return tiling_->totalPairSlots;
    }

    __aicore__ inline bool Resolve(uint64_t flatPair, CopyAddress &address)
    {
        const uint32_t batch = static_cast<uint32_t>(flatPair / tiling_->copyCap);
        const uint32_t copyIndex = static_cast<uint32_t>(
            flatPair - static_cast<uint64_t>(batch) * tiling_->copyCap);
        const uint64_t metadataOffset =
            static_cast<uint64_t>(batch) * tiling_->copyCap + copyIndex;
        const int32_t sourceToken = copySrcIdsGm_.GetValue(metadataOffset);
        const int32_t destinationSlot = copyDstSlotsGm_.GetValue(metadataOffset);
        ASSERT_MSG(sourceToken >= 0 && destinationSlot >= 0,
                   "A5 DSA fused LIDU emitted an invalid copy pair");
        if (sourceToken < 0 || destinationSlot < 0) {
            return false;
        }

        const uint32_t sourceBlockColumn =
            static_cast<uint32_t>(sourceToken) >> BLOCK_SHIFT;
        const uint32_t destinationBlockColumn =
            static_cast<uint32_t>(destinationSlot) >> BLOCK_SHIFT;
        ASSERT_MSG(sourceBlockColumn < tiling_->dramMaxBlockNum &&
                       destinationBlockColumn < tiling_->hbmMaxBlockNum,
                   "A5 DSA copy pair exceeds a block table");
        if (sourceBlockColumn >= tiling_->dramMaxBlockNum ||
            destinationBlockColumn >= tiling_->hbmMaxBlockNum) {
            return false;
        }
        const int32_t sourceBlock = dramBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->dramMaxBlockNum +
            sourceBlockColumn);
        const int32_t destinationBlock = hbmBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->hbmMaxBlockNum +
            destinationBlockColumn);
        ASSERT_MSG(
            sourceBlock >= 0 && destinationBlock >= 0 &&
                static_cast<uint32_t>(sourceBlock) <
                    tiling_->dramPhysicalBlockCount &&
                static_cast<uint32_t>(destinationBlock) <
                    tiling_->hbmPhysicalBlockCount,
            "A5 DSA copy pair resolves outside a physical cache");
        if (sourceBlock < 0 || destinationBlock < 0 ||
            static_cast<uint32_t>(sourceBlock) >=
                tiling_->dramPhysicalBlockCount ||
            static_cast<uint32_t>(destinationBlock) >=
                tiling_->hbmPhysicalBlockCount) {
            return false;
        }

        const uint64_t sourceRow =
            static_cast<uint64_t>(sourceBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(sourceToken) & BLOCK_MASK);
        const uint64_t destinationRow =
            static_cast<uint64_t>(destinationBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(destinationSlot) & BLOCK_MASK);
        address.sourceByteOffset = sourceRow * tiling_->packedRowBytes;
        address.destinationByteOffset =
            destinationRow * tiling_->packedRowBytes;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress &address)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyExtParams params{1, tiling_->packedRowBytes, 0, 0, 0};
        // dramKv may be backed by torch_npu.empty_with_swapped_memory.
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local, dramKvGm_[address.sourceByteOffset], params, pad);
        copyQueue_.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress &address)
    {
        LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams params{1, tiling_->packedRowBytes, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            hbmKvGm_[address.destinationByteOffset], local, params);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe *pipe_;
    const VllmA5KvcacheScatterCopyC8TilingData *tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t cachedBatch_ = static_cast<uint32_t>(-1);
    int32_t cachedCount_ = 0;
    GlobalTensor<uint8_t> hbmKvGm_;
    GlobalTensor<uint8_t> dramKvGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> copySrcIdsGm_;
    GlobalTensor<int32_t> copyDstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};
} // namespace

extern "C" __global__ __aicore__ void vllm_a5_kvcache_scatter_copy_c8(
    GM_ADDR hbmKv,
    GM_ADDR dramKv,
    GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable,
    GM_ADDR copySrcIds,
    GM_ADDR copyDstSlots,
    GM_ADDR copyCounts,
    GM_ADDR hbmKvOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)hbmKvOut;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(VllmA5KvcacheScatterCopyC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    VllmA5KvcacheScatterCopyC8Kernel op(&pipe, &tilingData);
    op.Init(
        hbmKv, dramKv, hbmBlockTable, dramBlockTable,
        copySrcIds, copyDstSlots, copyCounts);
    op.Process();
}
