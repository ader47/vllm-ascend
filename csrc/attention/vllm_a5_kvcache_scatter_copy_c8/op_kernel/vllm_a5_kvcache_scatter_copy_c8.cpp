/**
 * Ascend 950 token-granular copy for DSA C8 packed MLA cache rows.
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
constexpr uint32_t SPARSE_COUNT = 2048;
constexpr uint32_t RESIDENT_LENGTH_CHUNK = 256;

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
        GM_ADDR sourceTokenIds,
        GM_ADDR destinationSlots,
        GM_ADDR copyCounts,
        GM_ADDR cacheTokens,
        GM_ADDR candidateLens,
        GM_ADDR actualSeqLengthsKv,
        GM_ADDR attentionSlots,
        GM_ADDR residentSeqLengths)
    {
        coreIdx_ = GetBlockIdx();
        const uint32_t packedBufferBytes =
            (tiling_->packedRowBytes + 31U) & ~31U;
        pipe_->InitBuffer(copyQueue_, 2, packedBufferBytes);
        pipe_->InitBuffer(
            attentionSlotsBuf_,
            tiling_->attentionCapacity * sizeof(int32_t));
        pipe_->InitBuffer(
            residentLengthBuf_,
            RESIDENT_LENGTH_CHUNK * sizeof(int32_t));
        hbmKvGm_.SetGlobalBuffer((__gm__ uint8_t *)hbmKv);
        dramKvGm_.SetGlobalBuffer((__gm__ uint8_t *)dramKv);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)dramBlockTable);
        sourceTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t *)sourceTokenIds);
        destinationSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)destinationSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        actualSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsKv);
        attentionSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)attentionSlots);
        residentSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentSeqLengths);
    }

    __aicore__ inline void Process()
    {
        ValidateCopyCounts();
        BuildAttentionMetadata();
        // attention_slots rows are independent and stay batch-parallel.  The
        // compact resident_seq_lengths[B] output is published only by AIV0;
        // per-row writes from different AIVs false-share 32-byte GM lines.
        if (coreIdx_ == 0) {
            BuildResidentSeqLengths();
        }
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
    __aicore__ inline void ValidateCopyCounts()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            const int32_t count = copyCountsGm_.GetValue(batch);
            ASSERT_MSG(count >= 0,
                       "A5 DSA LI manager reported an invalid row");
            ASSERT_MSG(count <= static_cast<int32_t>(tiling_->copyCap),
                       "A5 DSA copy count exceeds metadata capacity");
        }
    }

    __aicore__ inline void BuildAttentionMetadata()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            LocalTensor<int32_t> slots = attentionSlotsBuf_.Get<int32_t>();
            Duplicate(slots, static_cast<int32_t>(-1),
                      tiling_->attentionCapacity);
            PipeBarrier<PIPE_ALL>();

            const int32_t cacheTokens = cacheTokensGm_.GetValue(batch);
            const int32_t candidateLen = candidateLensGm_.GetValue(batch);
            const int32_t actualLen = actualSeqLengthsKvGm_.GetValue(batch);
            if (cacheTokens > 0 && cacheTokens >= static_cast<int32_t>(SPARSE_COUNT) &&
                candidateLen >= static_cast<int32_t>(SPARSE_COUNT) &&
                actualLen >= candidateLen && actualLen - candidateLen <=
                    static_cast<int32_t>(BLOCK_SIZE)) {
                DataCopyExtParams topkCopy{
                    1,
                    static_cast<uint32_t>(SPARSE_COUNT * sizeof(int32_t)),
                    0,
                    0,
                    0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                const uint64_t topkOffset =
                    static_cast<uint64_t>(batch) * tiling_->copyCap;
                DataCopyPad<int32_t, PaddingMode::Normal>(
                    slots, destinationSlotsGm_[topkOffset], topkCopy, pad);
                PipeBarrier<PIPE_ALL>();
                const uint32_t tailCount =
                    static_cast<uint32_t>(actualLen - candidateLen);
                for (uint32_t index = 0; index < tailCount; ++index) {
                    slots.SetValue(
                        SPARSE_COUNT + index,
                        cacheTokens + static_cast<int32_t>(index));
                }
            } else if (cacheTokens == 0 && actualLen > 0) {
                // DENSE describes cache residency, not full-attention
                // computation.  The complete logical sequence is already in
                // HBM, while LI has selected at most SPARSE_COUNT logical
                // slots from it.  Forward those slots to QSFA instead of
                // materializing [0, actualLen), which would overflow the
                // fixed topK+tail metadata row once actualLen > 2176.
                const uint32_t denseTopkCount =
                    static_cast<uint32_t>(actualLen) < SPARSE_COUNT
                        ? static_cast<uint32_t>(actualLen)
                        : SPARSE_COUNT;
                DataCopyExtParams denseTopkCopy{
                    1,
                    static_cast<uint32_t>(denseTopkCount * sizeof(int32_t)),
                    0,
                    0,
                    0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                const uint64_t topkOffset =
                    static_cast<uint64_t>(batch) * tiling_->copyCap;
                DataCopyPad<int32_t, PaddingMode::Normal>(
                    slots, destinationSlotsGm_[topkOffset], denseTopkCopy, pad);
            }

            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams outputCopy{
                1,
                static_cast<uint32_t>(
                    tiling_->attentionCapacity * sizeof(int32_t)),
                0,
                0,
                0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                attentionSlotsGm_[
                    static_cast<uint64_t>(batch) * tiling_->attentionCapacity],
                slots,
                outputCopy);
        }
    }

    __aicore__ inline int32_t ResidentSeqLength(uint32_t batch)
    {
        const int32_t cacheTokens = cacheTokensGm_.GetValue(batch);
        const int32_t candidateLen = candidateLensGm_.GetValue(batch);
        const int32_t actualLen = actualSeqLengthsKvGm_.GetValue(batch);
        if (cacheTokens > 0 &&
            cacheTokens >= static_cast<int32_t>(SPARSE_COUNT) &&
            candidateLen >= static_cast<int32_t>(SPARSE_COUNT) &&
            actualLen >= candidateLen &&
            actualLen - candidateLen <= static_cast<int32_t>(BLOCK_SIZE)) {
            return cacheTokens + actualLen - candidateLen;
        }
        if (cacheTokens == 0 && actualLen >= 0) {
            return actualLen;
        }
        return 0;
    }

    __aicore__ inline void BuildResidentSeqLengths()
    {
        LocalTensor<int32_t> local = residentLengthBuf_.Get<int32_t>();
        for (uint32_t start = 0; start < tiling_->batchSize;
             start += RESIDENT_LENGTH_CHUNK) {
            const uint32_t remaining = tiling_->batchSize - start;
            const uint32_t count = remaining < RESIDENT_LENGTH_CHUNK
                ? remaining : RESIDENT_LENGTH_CHUNK;
            for (uint32_t index = 0; index < count; ++index) {
                local.SetValue(index, ResidentSeqLength(start + index));
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                residentSeqLengthsGm_[start], local, copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

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
        const int32_t sourceToken = sourceTokenIdsGm_.GetValue(metadataOffset);
        const int32_t destinationSlot = destinationSlotsGm_.GetValue(metadataOffset);
        ASSERT_MSG(sourceToken >= 0,
                   "A5 DSA source token is invalid within copy_count");
        ASSERT_MSG(destinationSlot >= 0,
                   "A5 DSA destination slot is invalid within copy_count");

        const uint32_t sourceBlockColumn =
            static_cast<uint32_t>(sourceToken) >> BLOCK_SHIFT;
        const uint32_t destinationBlockColumn =
            static_cast<uint32_t>(destinationSlot) >> BLOCK_SHIFT;
        ASSERT_MSG(sourceBlockColumn < tiling_->dramMaxBlockNum,
                   "A5 DSA source token exceeds DRAM block-table width");
        ASSERT_MSG(destinationBlockColumn < tiling_->hbmMaxBlockNum,
                   "A5 DSA destination slot exceeds HBM block-table width");
        const int32_t sourceBlock = dramBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->dramMaxBlockNum +
            sourceBlockColumn);
        const int32_t destinationBlock = hbmBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->hbmMaxBlockNum +
            destinationBlockColumn);
        ASSERT_MSG(sourceBlock >= 0,
                   "A5 DSA source block-table entry is invalid");
        ASSERT_MSG(destinationBlock >= 0,
                   "A5 DSA destination block-table entry is invalid");
        ASSERT_MSG(
            sourceBlock < static_cast<int32_t>(
                tiling_->dramPhysicalBlockCount),
            "A5 DSA source physical block exceeds DRAM capacity");
        ASSERT_MSG(
            destinationBlock < static_cast<int32_t>(
                tiling_->hbmPhysicalBlockCount),
            "A5 DSA destination physical block exceeds HBM capacity");

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
    GlobalTensor<int32_t> sourceTokenIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> actualSeqLengthsKvGm_;
    GlobalTensor<int32_t> attentionSlotsGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
    TBuf<TPosition::VECCALC> attentionSlotsBuf_;
    TBuf<TPosition::VECCALC> residentLengthBuf_;
};
} // namespace

extern "C" __global__ __aicore__ void vllm_a5_kvcache_scatter_copy_c8(
    GM_ADDR hbmKv,
    GM_ADDR dramKv,
    GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable,
    GM_ADDR sourceTokenIds,
    GM_ADDR destinationSlots,
    GM_ADDR copyCounts,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR actualSeqLengthsKv,
    GM_ADDR hbmKvOut,
    GM_ADDR attentionSlotsOut,
    GM_ADDR residentSeqLengthsOut,
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
        sourceTokenIds, destinationSlots, copyCounts,
        cacheTokens, candidateLens, actualSeqLengthsKv,
        attentionSlotsOut, residentSeqLengthsOut);
    op.Process();
}
