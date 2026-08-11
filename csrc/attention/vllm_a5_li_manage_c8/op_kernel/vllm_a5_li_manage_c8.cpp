/**
 * AIV-only request-pool manager for the native A5 C8 QuantLightningIndexer.
 * Scoring/top-k is deliberately not duplicated here: the official operator
 * supplies the exact C8 top-2048 set, while this kernel owns offload state.
 */

#include "kernel_operator.h"
#include "vllm_a5_li_manage_c8_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t TOPK = 2048;
constexpr uint32_t OUTPUT_CAPACITY = 16384;
constexpr uint32_t CACHE_CHUNK = 2048;
constexpr uint32_t MAX_CACHE_TOKENS = 12288;
constexpr uint32_t HASH_CAPACITY = 4096;
constexpr uint32_t HASH_MASK = HASH_CAPACITY - 1;
constexpr uint32_t MTE_SCALAR_ALIGN_INTS = 8;
constexpr int32_t INVALID = -1;
constexpr int32_t ROW_MODE_PAD = 0;
constexpr int32_t ROW_MODE_DENSE = 1;
constexpr int32_t ROW_MODE_SPARSE = 2;

template <HardEvent event>
__aicore__ inline void SyncPipes()
{
    const event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

class VllmA5LiManageC8Kernel {
public:
    __aicore__ inline VllmA5LiManageC8Kernel(
        TPipe *pipe, const VllmA5LiManageC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR topkIndices, GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsPool, GM_ADDR rowModes,
        GM_ADDR actualSeqLengthsKey, GM_ADDR sourceIds,
        GM_ADDR destinationSlots, GM_ADDR missCounts,
        GM_ADDR tailInfo)
    {
        coreIdx_ = GetBlockIdx();
        topkIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsPoolGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        actualSeqLengthsKeyGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsKey);
        sourceIdsGm_.SetGlobalBuffer((__gm__ int32_t *)sourceIds);
        destinationSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)destinationSlots);
        missCountsGm_.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        tailInfoGm_.SetGlobalBuffer((__gm__ int32_t *)tailInfo);

        pipe_->InitBuffer(topkBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(slotBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(missTokenBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(hitTokenBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(hitSlotBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(cacheChunkBuf_, CACHE_CHUNK * sizeof(int32_t));
        pipe_->InitBuffer(protectedSlotBuf_, MAX_CACHE_TOKENS * sizeof(uint8_t));
        pipe_->InitBuffer(hashBuf_, HASH_CAPACITY * sizeof(uint32_t));
        // MTE3 requires each UB source address to be 32-byte aligned.  Keep
        // miss_count at lane 0 and the two-element tail_info at lane 8.
        pipe_->InitBuffer(scalarBuf_, 64);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            ProcessRow(batch);
        }
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline void FillGlobalRange(
        GlobalTensor<int32_t> destination, uint32_t count,
        int32_t value, LocalTensor<int32_t> scratch)
    {
        if (count == 0) {
            return;
        }

        // InitGlobalMemory may only be used before TPipe::InitBuffer.  Runtime
        // state is initialized through the already allocated UB scratchpad.
        // scratch may have been written through PIPE_S by first-fill.  Make
        // that dependency explicit before reusing the same UB as a vector
        // fill source.
        SyncPipes<HardEvent::S_V>();
        Duplicate(scratch, value, CACHE_CHUNK);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
        for (uint32_t offset = 0; offset < count; offset += CACHE_CHUNK) {
            const uint32_t chunkLen = MinU32(CACHE_CHUNK, count - offset);
            DataCopyExtParams copy{
                1, static_cast<uint32_t>(chunkLen * sizeof(int32_t)),
                0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                destination[offset], scratch, copy);
            SyncPipes<HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline void ClearOutputTail(
        uint32_t batch, uint32_t validCount)
    {
        if (validCount >= tiling_->outputCapacity) {
            return;
        }
        const uint64_t base =
            static_cast<uint64_t>(batch) * tiling_->outputCapacity +
            validCount;
        const uint32_t count = tiling_->outputCapacity - validCount;
        GlobalTensor<int32_t> sourceTail = sourceIdsGm_[base];
        GlobalTensor<int32_t> slotTail = destinationSlotsGm_[base];
        LocalTensor<int32_t> scratch = cacheChunkBuf_.Get<int32_t>();
        FillGlobalRange(sourceTail, count, INVALID, scratch);
        FillGlobalRange(slotTail, count, INVALID, scratch);
    }

    __aicore__ inline void StoreMetadata(
        uint32_t batch, int32_t missCount,
        int32_t tailSlotStart, int32_t tailTokenCount)
    {
        LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
        scalar.SetValue(0, missCount);
        scalar.SetValue(MTE_SCALAR_ALIGN_INTS, tailSlotStart);
        scalar.SetValue(MTE_SCALAR_ALIGN_INTS + 1, tailTokenCount);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::S_MTE3>();
        DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyParams two{1, static_cast<uint16_t>(2 * sizeof(int32_t)), 0, 0};
        DataCopyPad(missCountsGm_[batch], scalar, one);
        DataCopyPad(
            tailInfoGm_[static_cast<uint64_t>(batch) * 2],
            scalar[MTE_SCALAR_ALIGN_INTS], two);
        SyncPipes<HardEvent::MTE3_S>();
    }

    __aicore__ inline void WritePoolScalar(
        uint64_t offset, int32_t value)
    {
        // TOKEN_CAPACITY+1 makes neighbouring request rows share a DCache
        // line.  Scalar MTE3 writes avoid stale cross-row writeback.
        LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
        scalar.SetValue(0, value);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::S_MTE3>();
        DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(cacheSlotsPoolGm_[offset], scalar, one);
        SyncPipes<HardEvent::MTE3_S>();
    }

    __aicore__ inline void LoadTopk(
        uint32_t batch, LocalTensor<int32_t> topk)
    {
        DataCopyExtParams copy{1, TOPK * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(
            topk,
            topkIndicesGm_[static_cast<uint64_t>(batch) * TOPK],
            copy, pad);
        SyncPipes<HardEvent::MTE2_S>();
    }

    __aicore__ inline void StoreRange(
        GlobalTensor<int32_t> destination,
        LocalTensor<int32_t> source, uint32_t count)
    {
        if (count == 0) {
            return;
        }
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        SyncPipes<HardEvent::S_MTE3>();
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destination, source, copy);
        SyncPipes<HardEvent::MTE3_S>();
    }

    __aicore__ inline void StoreTopk(
        uint32_t batch, LocalTensor<int32_t> tokens,
        LocalTensor<int32_t> slots)
    {
        const uint64_t base =
            static_cast<uint64_t>(batch) * tiling_->outputCapacity;
        StoreRange(sourceIdsGm_[base], tokens, TOPK);
        StoreRange(destinationSlotsGm_[base], slots, TOPK);
    }

    __aicore__ inline bool HashContains(
        LocalTensor<uint32_t> hash, uint32_t token) const
    {
        uint32_t position = (token * 2654435761U) & HASH_MASK;
        while (true) {
            const uint32_t current = hash.GetValue(position);
            if (current == token) {
                return true;
            }
            if (current == 0xffffffffU) {
                return false;
            }
            position = (position + 1U) & HASH_MASK;
        }
    }

    __aicore__ inline void HashInsert(
        LocalTensor<uint32_t> hash, uint32_t token)
    {
        uint32_t position = (token * 2654435761U) & HASH_MASK;
        while (hash.GetValue(position) != 0xffffffffU &&
               hash.GetValue(position) != token) {
            position = (position + 1U) & HASH_MASK;
        }
        hash.SetValue(position, token);
    }

    __aicore__ inline void ProcessDense(
        uint32_t batch, LocalTensor<int32_t> topk,
        LocalTensor<int32_t> slots)
    {
        LoadTopk(batch, topk);
        for (uint32_t index = 0; index < TOPK; ++index) {
            slots.SetValue(index, topk.GetValue(index));
        }
        PipeBarrier<PIPE_V>();
        StoreTopk(batch, topk, slots);
        ClearOutputTail(batch, TOPK);
        StoreMetadata(batch, 0, INVALID, 0);
    }

    __aicore__ inline bool ValidateSparse(
        int32_t poolRow, int32_t actualLen, int32_t metadata,
        uint32_t &sourceLen, uint32_t &tailLen,
        uint32_t &budget, bool &firstFill) const
    {
        if (poolRow < 0 ||
            poolRow >= static_cast<int32_t>(tiling_->poolSize) ||
            actualLen < static_cast<int32_t>(TOPK) ||
            actualLen > static_cast<int32_t>(tiling_->tokenCapacity) ||
            metadata == 0) {
            return false;
        }
        const uint32_t actual = static_cast<uint32_t>(actualLen);
        // Keep the final physical block resident as the dense tail.  This is
        // intentionally (actual - 1) / 128 rather than actual / 128 so an
        // exact block boundary produces tailLen=128.
        sourceLen = (actual - 1U) / BLOCK_SIZE * BLOCK_SIZE;
        tailLen = actual - sourceLen;
        firstFill = metadata < 0;
        budget = static_cast<uint32_t>(firstFill ? -metadata : metadata);
        return sourceLen >= TOPK && budget >= TOPK &&
            budget <= MAX_CACHE_TOKENS && budget <= sourceLen;
    }

    __aicore__ inline void ProcessFirstFill(
        uint32_t batch, uint64_t poolBase,
        uint32_t sourceLen, uint32_t tailLen, uint32_t budget,
        LocalTensor<int32_t> topk, LocalTensor<int32_t> slots,
        LocalTensor<uint32_t> hash)
    {
        GlobalTensor<int32_t> poolTokenRow = cacheSlotsPoolGm_[poolBase];
        // Use the colleague-validated hash first-fill as the correctness
        // baseline.  Membership is intentionally independent of the 2048
        // element cache-copy chunk boundary.  Initialization remains on the
        // integrated DMA path:
        // InitGlobalMemory cannot be used after TPipe buffer allocation, and
        // adjacent request rows are not cache-line aligned.
        FillGlobalRange(
            poolTokenRow, tiling_->tokenCapacity, INVALID,
            cacheChunkBuf_.Get<int32_t>());

        LoadTopk(batch, topk);
        Duplicate(hash, 0xffffffffU, HASH_CAPACITY);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_S>();
        for (uint32_t index = 0; index < TOPK; ++index) {
            const int32_t token = topk.GetValue(index);
            slots.SetValue(index, static_cast<int32_t>(index));
            if (token >= 0 && token < static_cast<int32_t>(sourceLen)) {
                HashInsert(hash, static_cast<uint32_t>(token));
                WritePoolScalar(
                    poolBase + static_cast<uint32_t>(token),
                    static_cast<int32_t>(index));
            }
        }
        StoreTopk(batch, topk, slots);

        const uint32_t validCount = MinU32(sourceLen, OUTPUT_CAPACITY);
        uint32_t outputPosition = TOPK;
        uint32_t token = 0;
        while (outputPosition < validCount) {
            uint32_t chunkCount = 0;
            while (token < sourceLen && chunkCount < CACHE_CHUNK &&
                   outputPosition + chunkCount < validCount) {
                if (!HashContains(hash, token)) {
                    topk.SetValue(chunkCount, static_cast<int32_t>(token));
                    slots.SetValue(
                        chunkCount,
                        static_cast<int32_t>(outputPosition + chunkCount));
                    if (outputPosition + chunkCount < budget) {
                        WritePoolScalar(
                            poolBase + token,
                            static_cast<int32_t>(outputPosition + chunkCount));
                    }
                    ++chunkCount;
                }
                ++token;
            }
            const uint64_t outputBase =
                static_cast<uint64_t>(batch) * tiling_->outputCapacity +
                outputPosition;
            StoreRange(sourceIdsGm_[outputBase], topk, chunkCount);
            StoreRange(destinationSlotsGm_[outputBase], slots, chunkCount);
            outputPosition += chunkCount;
        }
        ClearOutputTail(batch, validCount);
        WritePoolScalar(poolBase + tiling_->tokenCapacity,
                        static_cast<int32_t>(budget));
        StoreMetadata(
            batch, static_cast<int32_t>(budget),
            static_cast<int32_t>(budget),
            static_cast<int32_t>(tailLen));
    }

    __aicore__ inline bool ProcessSteady(
        uint32_t batch, uint64_t poolBase,
        uint32_t sourceLen, uint32_t tailLen, uint32_t budget,
        LocalTensor<int32_t> topk, LocalTensor<int32_t> slots,
        LocalTensor<int32_t> missTokens,
        LocalTensor<int32_t> hitTokens,
        LocalTensor<int32_t> hitSlots,
        LocalTensor<int32_t> cacheChunk,
        LocalTensor<uint8_t> protectedSlots)
    {
        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_S>();
        LoadTopk(batch, topk);

        uint32_t missCount = 0;
        uint32_t hitCount = 0;
        for (uint32_t index = 0; index < TOPK; ++index) {
            const int32_t token = topk.GetValue(index);
            int32_t slot = INVALID;
            if (token >= 0 && token < static_cast<int32_t>(sourceLen)) {
                slot = cacheSlotsPoolGm_.GetValue(
                    poolBase + static_cast<uint32_t>(token));
            }
            if (slot >= 0 && slot < static_cast<int32_t>(budget)) {
                hitTokens.SetValue(hitCount, token);
                hitSlots.SetValue(hitCount, slot);
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++hitCount;
            } else {
                missTokens.SetValue(missCount, token);
                ++missCount;
            }
        }

        uint32_t assigned = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < sourceLen && assigned < missCount;
             chunkBase += CACHE_CHUNK) {
            const uint32_t chunkLen =
                MinU32(CACHE_CHUNK, sourceLen - chunkBase);
            DataCopyExtParams copy{
                1, static_cast<uint32_t>(chunkLen * sizeof(int32_t)),
                0, 0, 0};
            DataCopyPad(
                cacheChunk,
                cacheSlotsPoolGm_[poolBase + chunkBase], copy, pad);
            SyncPipes<HardEvent::MTE2_S>();
            for (uint32_t offset = 0;
                 offset < chunkLen && assigned < missCount; ++offset) {
                const int32_t slot = cacheChunk.GetValue(offset);
                if (slot < 0 || slot >= static_cast<int32_t>(budget) ||
                    protectedSlots.GetValue(static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                const int32_t missToken = missTokens.GetValue(assigned);
                if (missToken < 0 ||
                    missToken >= static_cast<int32_t>(sourceLen)) {
                    return false;
                }
                const uint32_t victimToken = chunkBase + offset;
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 2);
                WritePoolScalar(poolBase + victimToken, INVALID);
                WritePoolScalar(
                    poolBase + static_cast<uint32_t>(missToken), slot);
                topk.SetValue(assigned, missToken);
                slots.SetValue(assigned, slot);
                ++assigned;
            }
        }
        if (assigned != missCount || missCount + hitCount != TOPK) {
            return false;
        }
        for (uint32_t index = 0; index < hitCount; ++index) {
            topk.SetValue(missCount + index, hitTokens.GetValue(index));
            slots.SetValue(missCount + index, hitSlots.GetValue(index));
        }
        PipeBarrier<PIPE_V>();
        StoreTopk(batch, topk, slots);
        ClearOutputTail(batch, TOPK);
        StoreMetadata(
            batch, static_cast<int32_t>(missCount),
            static_cast<int32_t>(budget),
            static_cast<int32_t>(tailLen));
        return true;
    }

    __aicore__ inline void ProcessRow(uint32_t batch)
    {
        const int32_t mode = rowModesGm_.GetValue(batch);
        if (mode == ROW_MODE_PAD) {
            ClearOutputTail(batch, 0);
            StoreMetadata(batch, 0, INVALID, 0);
            return;
        }

        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        if (mode == ROW_MODE_DENSE) {
            ProcessDense(batch, topk, slots);
            return;
        }
        if (mode != ROW_MODE_SPARSE) {
            ClearOutputTail(batch, 0);
            StoreMetadata(batch, -1, INVALID, 0);
            return;
        }

        const int32_t poolRow = reqPoolEntriesGm_.GetValue(batch);
        const int32_t actualLen = actualSeqLengthsKeyGm_.GetValue(batch);
        if (poolRow < 0 ||
            poolRow >= static_cast<int32_t>(tiling_->poolSize)) {
            ClearOutputTail(batch, 0);
            StoreMetadata(batch, -1, INVALID, 0);
            return;
        }
        const uint64_t poolBase =
            static_cast<uint64_t>(poolRow) *
            (static_cast<uint64_t>(tiling_->tokenCapacity) + 1U);
        const int32_t metadata = cacheSlotsPoolGm_.GetValue(
            poolBase + tiling_->tokenCapacity);
        uint32_t sourceLen = 0;
        uint32_t tailLen = 0;
        uint32_t budget = 0;
        bool firstFill = false;
        if (!ValidateSparse(
                poolRow, actualLen, metadata,
                sourceLen, tailLen, budget, firstFill)) {
            ClearOutputTail(batch, 0);
            StoreMetadata(batch, -1, INVALID, 0);
            return;
        }

        if (firstFill) {
            ProcessFirstFill(
                batch, poolBase, sourceLen, tailLen, budget,
                topk, slots, hashBuf_.Get<uint32_t>());
            return;
        }
        const bool ok = ProcessSteady(
            batch, poolBase, sourceLen, tailLen, budget,
            topk, slots, missTokenBuf_.Get<int32_t>(),
            hitTokenBuf_.Get<int32_t>(), hitSlotBuf_.Get<int32_t>(),
            cacheChunkBuf_.Get<int32_t>(),
            protectedSlotBuf_.Get<uint8_t>());
        if (!ok) {
            ClearOutputTail(batch, 0);
            StoreMetadata(batch, -1, INVALID, 0);
        }
    }

private:
    TPipe *pipe_;
    const VllmA5LiManageC8TilingData *tiling_;
    uint32_t coreIdx_ = 0;
    GlobalTensor<int32_t> topkIndicesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsPoolGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> actualSeqLengthsKeyGm_;
    GlobalTensor<int32_t> sourceIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    GlobalTensor<int32_t> tailInfoGm_;
    TBuf<TPosition::VECCALC> topkBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TBuf<TPosition::VECCALC> missTokenBuf_;
    TBuf<TPosition::VECCALC> hitTokenBuf_;
    TBuf<TPosition::VECCALC> hitSlotBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
    TBuf<TPosition::VECCALC> protectedSlotBuf_;
    TBuf<TPosition::VECCALC> hashBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};
} // namespace

extern "C" __global__ __aicore__ void vllm_a5_li_manage_c8(
    GM_ADDR topkIndices, GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool, GM_ADDR rowModes,
    GM_ADDR actualSeqLengthsKey, GM_ADDR sourceIds,
    GM_ADDR destinationSlots, GM_ADDR missCounts,
    GM_ADDR tailInfo, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(VllmA5LiManageC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    VllmA5LiManageC8Kernel op(&pipe, &tilingData);
    op.Init(
        topkIndices, reqPoolEntries, cacheSlotsPool, rowModes,
        actualSeqLengthsKey, sourceIds, destinationSlots,
        missCounts, tailInfo);
    op.Process();
}
