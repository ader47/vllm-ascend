/**
 * Correctness-first request-pool phase of the fused MTP operator.
 *
 * One AIV processes one request and its 1..4 TND query rows.  It builds the
 * unique union of those rows, updates the persistent source-token -> HBM-slot
 * mapping once, publishes one copy list per request, and publishes the full
 * top-2048 + causal-tail sparse index row consumed by the native A5 C8 SFA.
 */

#include "kernel_operator.h"
#include "vllm_a5_li_manage_c8_tiling.h"

#ifndef VLLM_A5_LI_MANAGE_C8_MANAGER_H
#define VLLM_A5_LI_MANAGE_C8_MANAGER_H

namespace vllm_a5_mtp_manager {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t TOPK = 2048;
constexpr uint32_t TAIL_CAPACITY = 2 * BLOCK_SIZE;
constexpr uint32_t TAIL_BLOCK_COUNT = 2;
constexpr uint32_t TAIL_STORAGE_CAPACITY =
    TAIL_BLOCK_COUNT * BLOCK_SIZE;
constexpr uint32_t ATTENTION_CAPACITY = TOPK + TAIL_CAPACITY;
constexpr uint32_t OUTPUT_CAPACITY = 16384;
constexpr uint32_t MAX_QUERIES_PER_REQUEST = 4;
constexpr uint32_t CACHE_CHUNK = 2048;
constexpr uint32_t MAX_CACHE_TOKENS = 12288;
constexpr uint32_t HASH_CAPACITY = 16384;
constexpr uint32_t HASH_MASK = HASH_CAPACITY - 1;
// One 32-bit hash payload packs {slot14, token18}; MISS_SLOT is outside every
// supported C, and HASH_EMPTY stays distinct even for token 2^18-1.
constexpr uint32_t TOKEN_MASK = (1U << 18) - 1U;
constexpr uint32_t SLOT_SHIFT = 18;
// 14 slot bits can encode [0, 16383]. 12288 is a legal first slot in the
// largest budget's parity-0 tail, so it cannot also serve as the miss marker.
constexpr uint32_t MISS_SLOT = 0x3ffeU;
constexpr uint32_t HASH_EMPTY = 0xffffffffU;
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
        GM_ADDR topkIndices, GM_ADDR actualSeqLengthsQuery,
        GM_ADDR candidateLens, GM_ADDR finalSeqLengthsKv,
        GM_ADDR rowModes, GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsPool, GM_ADDR sparseAndTailSlots,
        GM_ADDR sparseAndTailSrcIds, GM_ADDR perQueryMissCounts,
        GM_ADDR residentSeqLengths, GM_ADDR copySrcIds,
        GM_ADDR copyDstSlots, GM_ADDR copyCounts,
        uint32_t coreIdx, uint32_t topkRowStride)
    {
        coreIdx_ = coreIdx;
        topkRowStride_ = topkRowStride;
        topkIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        finalSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)finalSeqLengthsKv);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsPoolGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        sparseAndTailSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sparseAndTailSlots);
        sparseAndTailSrcIdsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sparseAndTailSrcIds);
        perQueryMissCountsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)perQueryMissCounts);
        residentSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentSeqLengths);
        copySrcIdsGm_.SetGlobalBuffer((__gm__ int32_t *)copySrcIds);
        copyDstSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)copyDstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);

        pipe_->InitBuffer(topkBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(slotBuf_, ATTENTION_CAPACITY * sizeof(int32_t));
        pipe_->InitBuffer(sourceBuf_, ATTENTION_CAPACITY * sizeof(int32_t));
        pipe_->InitBuffer(cacheChunkBuf_, CACHE_CHUNK * sizeof(int32_t));
        pipe_->InitBuffer(protectedSlotBuf_, MAX_CACHE_TOKENS * sizeof(uint8_t));
        pipe_->InitBuffer(hashBuf_, HASH_CAPACITY * sizeof(uint32_t));
        // MTE3 scalar sources must remain 32-byte aligned.
        pipe_->InitBuffer(scalarBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            ProcessRequest(batch);
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

        // InitGlobalMemory is only valid before TPipe::InitBuffer. Runtime
        // request state is therefore cleared through the existing UB scratch.
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

    __aicore__ inline uint32_t PackHashEntry(
        uint32_t token, uint32_t slot) const
    {
        return (slot << SLOT_SHIFT) | token;
    }

    __aicore__ inline void InitHash(LocalTensor<uint32_t> hash)
    {
        Duplicate(hash, HASH_EMPTY, HASH_CAPACITY);
        // FindHashPosition immediately consumes the vector-written hash from
        // the scalar pipeline. PIPE_V alone does not establish that ordering.
        SyncPipes<HardEvent::V_S>();
    }

    __aicore__ inline bool FindHashPosition(
        LocalTensor<uint32_t> hash, int32_t token,
        uint32_t &position, bool &found) const
    {
        position =
            (static_cast<uint32_t>(token) * 2654435761U) & HASH_MASK;
        for (uint32_t probe = 0; probe < HASH_CAPACITY; ++probe) {
            const uint32_t entry = hash.GetValue(position);
            if (entry != HASH_EMPTY &&
                (entry & TOKEN_MASK) == static_cast<uint32_t>(token)) {
                found = true;
                return true;
            }
            if (entry == HASH_EMPTY) {
                found = false;
                return true;
            }
            position = (position + 1U) & HASH_MASK;
        }
        found = false;
        return false;
    }

    __aicore__ inline void WritePoolScalar(uint64_t offset, int32_t value)
    {
        LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
        scalar.SetValue(0, value);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(cacheSlotsPoolGm_[offset], scalar, one);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }

    __aicore__ inline void WriteOutputScalar(
        GlobalTensor<int32_t> destination, int32_t value)
    {
        LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
        scalar.SetValue(0, value);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(destination, scalar, one);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void LoadTopk(
        uint32_t queryRow, LocalTensor<int32_t> topk)
    {
        DataCopyExtParams copy{1, TOPK * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(
            topk,
            topkIndicesGm_[
                static_cast<uint64_t>(queryRow) * topkRowStride_],
            copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
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
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        DataCopyPad<int32_t, PaddingMode::Normal>(destination, source, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void InitAttentionRows(
        LocalTensor<int32_t> slots, LocalTensor<int32_t> sources)
    {
        Duplicate(slots, INVALID, ATTENTION_CAPACITY);
        Duplicate(sources, INVALID, ATTENTION_CAPACITY);
        // The following scalar SetValue calls selectively overwrite this
        // vector fill. Explicit V->S ordering prevents the fill from racing
        // with those writes and leaving a partially initialized output row.
        SyncPipes<HardEvent::V_S>();
    }

    __aicore__ inline void StoreAttentionRows(
        uint32_t queryRow, LocalTensor<int32_t> slots,
        LocalTensor<int32_t> sources)
    {
        const uint64_t offset =
            static_cast<uint64_t>(queryRow) * ATTENTION_CAPACITY;
        StoreRange(sparseAndTailSlotsGm_[offset], slots, ATTENTION_CAPACITY);
        StoreRange(
            sparseAndTailSrcIdsGm_[offset], sources, ATTENTION_CAPACITY);
    }

    __aicore__ inline void ClearAttentionRows(
        uint32_t queryRow, int32_t missCount)
    {
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        InitAttentionRows(slots, sources);
        StoreAttentionRows(queryRow, slots, sources);
        WriteOutputScalar(perQueryMissCountsGm_[queryRow], missCount);
    }

    __aicore__ inline bool AppendCausalTail(
        uint32_t queryRow, uint32_t queryEnd,
        uint32_t sourceLen, uint32_t finalLen, uint32_t budget,
        LocalTensor<int32_t> slots, LocalTensor<int32_t> sources)
    {
        const uint32_t laterQueries = queryEnd - 1U - queryRow;
        if (finalLen <= laterQueries) {
            return false;
        }
        const uint32_t visibleLen = finalLen - laterQueries;
        const uint32_t tailTokenStart = sourceLen;
        if (visibleLen < tailTokenStart) {
            return false;
        }
        const uint32_t tailCount = visibleLen - tailTokenStart;
        if (tailCount > TAIL_CAPACITY) {
            return false;
        }
        for (uint32_t index = 0; index < tailCount; ++index) {
            const uint32_t token = tailTokenStart + index;
            slots.SetValue(
                TOPK + index,
                static_cast<int32_t>(
                    budget + token % TAIL_STORAGE_CAPACITY));
            sources.SetValue(TOPK + index, static_cast<int32_t>(token));
        }
        PipeBarrier<PIPE_V>();
        StoreAttentionRows(queryRow, slots, sources);
        return true;
    }

    __aicore__ inline bool PublishSparseSelection(
        uint32_t queryRow, uint32_t queryEnd, uint32_t sourceLen,
        uint32_t finalLen, uint32_t budget, bool firstFill,
        LocalTensor<int32_t> topk, LocalTensor<int32_t> slots,
        LocalTensor<int32_t> sources, LocalTensor<uint8_t> protectedSlots,
        LocalTensor<uint32_t> hash)
    {
        LoadTopk(queryRow, topk);
        uint32_t missCount = 0U;
        for (uint32_t index = 0U; index < TOPK; ++index) {
            const int32_t token = topk.GetValue(index);
            uint32_t hashPosition = 0U;
            bool found = false;
            if (!FindHashPosition(hash, token, hashPosition, found) || !found) {
                return false;
            }
            const uint32_t slot = hash.GetValue(hashPosition) >> SLOT_SHIFT;
            if (token < 0 || static_cast<uint32_t>(token) >= sourceLen ||
                slot >= budget) {
                return false;
            }
            const bool isMiss = firstFill || protectedSlots.GetValue(slot) == 2U;
            missCount += isMiss ? 1U : 0U;
        }

        InitAttentionRows(slots, sources);
        uint32_t missCursor = 0U;
        uint32_t hitCursor = missCount;
        for (uint32_t index = 0U; index < TOPK; ++index) {
            const int32_t token = topk.GetValue(index);
            uint32_t hashPosition = 0U;
            bool found = false;
            if (!FindHashPosition(hash, token, hashPosition, found) || !found) {
                return false;
            }
            const uint32_t slot = hash.GetValue(hashPosition) >> SLOT_SHIFT;
            const bool isMiss = firstFill || protectedSlots.GetValue(slot) == 2U;
            const uint32_t output = isMiss ? missCursor++ : hitCursor++;
            sources.SetValue(output, token);
            slots.SetValue(output, static_cast<int32_t>(slot));
        }
        PipeBarrier<PIPE_V>();
        WriteOutputScalar(
            perQueryMissCountsGm_[queryRow],
            static_cast<int32_t>(missCount));
        return AppendCausalTail(
            queryRow, queryEnd, sourceLen, finalLen, budget, slots, sources);
    }

    __aicore__ inline void ClearCopyRow(uint32_t batch)
    {
        const uint64_t base =
            static_cast<uint64_t>(batch) * tiling_->outputCapacity;
        GlobalTensor<int32_t> sourceOutput = copySrcIdsGm_[base];
        GlobalTensor<int32_t> destinationOutput = copyDstSlotsGm_[base];
        LocalTensor<int32_t> scratch = cacheChunkBuf_.Get<int32_t>();
        FillGlobalRange(
            sourceOutput, tiling_->outputCapacity, INVALID, scratch);
        FillGlobalRange(
            destinationOutput, tiling_->outputCapacity, INVALID, scratch);
    }

    __aicore__ inline void WriteCopyPair(
        uint32_t batch, uint32_t index, int32_t token, int32_t slot)
    {
        const uint64_t offset =
            static_cast<uint64_t>(batch) * tiling_->outputCapacity + index;
        WriteOutputScalar(copySrcIdsGm_[offset], token);
        WriteOutputScalar(copyDstSlotsGm_[offset], slot);
    }

    __aicore__ inline void PublishDenseSelectionError(
        uint32_t batch, uint32_t queryRow, uint32_t visibleLen,
        LocalTensor<int32_t> topk)
    {
        const uint64_t base =
            static_cast<uint64_t>(batch) * tiling_->outputCapacity;
        StoreRange(copySrcIdsGm_[base], topk, TOPK);
        WriteOutputScalar(
            copyDstSlotsGm_[base], static_cast<int32_t>(queryRow));
        WriteOutputScalar(
            copyDstSlotsGm_[base + 1U],
            static_cast<int32_t>(visibleLen));
    }

    __aicore__ inline bool QueryRange(
        uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd) const
    {
        const int32_t end = actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t start = batch == 0
            ? 0 : actualSeqLengthsQueryGm_.GetValue(batch - 1);
        if (start < 0 || end <= start ||
            end > static_cast<int32_t>(tiling_->totalQueryRows) ||
            end - start > static_cast<int32_t>(MAX_QUERIES_PER_REQUEST)) {
            return false;
        }
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        return true;
    }

    __aicore__ inline void MarkRequestError(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd)
    {
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            ClearAttentionRows(queryRow, -1);
        }
        WriteOutputScalar(residentSeqLengthsGm_[batch], 0);
        WriteOutputScalar(copyCountsGm_[batch], -1);
    }

    __aicore__ inline void ProcessPad(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd)
    {
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            ClearAttentionRows(queryRow, 0);
        }
        WriteOutputScalar(residentSeqLengthsGm_[batch], 0);
        WriteOutputScalar(copyCountsGm_[batch], 0);
    }

    __aicore__ inline bool ProcessDense(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        int32_t finalLen,
        LocalTensor<int32_t> topk, LocalTensor<int32_t> slots,
        LocalTensor<int32_t> sources)
    {
        const uint32_t queryCount = queryEnd - queryStart;
        if (finalLen < static_cast<int32_t>(queryCount) ||
            finalLen > static_cast<int32_t>(tiling_->tokenCapacity)) {
            return false;
        }
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            const uint32_t laterQueries = queryEnd - 1U - queryRow;
            if (static_cast<uint32_t>(finalLen) <= laterQueries) {
                return false;
            }
            const uint32_t visibleLen =
                static_cast<uint32_t>(finalLen) - laterQueries;
            if (visibleLen > tiling_->maxCandidateLen) {
                return false;
            }
            InitAttentionRows(slots, sources);
            if (visibleLen <= TOPK) {
                if (visibleLen > ATTENTION_CAPACITY) {
                    return false;
                }
                for (uint32_t index = 0; index < visibleLen; ++index) {
                    slots.SetValue(index, static_cast<int32_t>(index));
                    sources.SetValue(index, static_cast<int32_t>(index));
                }
            } else {
                LoadTopk(queryRow, topk);
                for (uint32_t index = 0; index < TOPK; ++index) {
                    const int32_t token = topk.GetValue(index);
                    if (token < 0 ||
                        token >= static_cast<int32_t>(visibleLen)) {
                        PublishDenseSelectionError(
                            batch, queryRow, visibleLen, topk);
                        return false;
                    }
                    // DENSE 行的完整 KV 已在 HBM；逻辑 token 位置就是
                    // QSFA 消费的 resident slot，不产生 DRAM->HBM IO。
                    slots.SetValue(index, token);
                    sources.SetValue(index, token);
                }
            }
            PipeBarrier<PIPE_V>();
            StoreAttentionRows(queryRow, slots, sources);
            WriteOutputScalar(perQueryMissCountsGm_[queryRow], 0);
        }
        WriteOutputScalar(residentSeqLengthsGm_[batch], finalLen);
        WriteOutputScalar(copyCountsGm_[batch], 0);
        return true;
    }

    __aicore__ inline bool ValidateSparse(
        int32_t poolRow, int32_t candidateLen, int32_t finalLen,
        int32_t metadata, uint32_t queryCount, uint32_t &sourceLen,
        uint32_t &budget, bool &firstFill) const
    {
        if (poolRow < 0 ||
            poolRow >= static_cast<int32_t>(tiling_->poolSize) ||
            candidateLen < static_cast<int32_t>(TOPK) ||
            candidateLen > static_cast<int32_t>(tiling_->tokenCapacity) ||
            candidateLen % static_cast<int32_t>(BLOCK_SIZE) != 0 ||
            finalLen < static_cast<int32_t>(queryCount) ||
            candidateLen > finalLen ||
            finalLen - candidateLen >
                static_cast<int32_t>(TAIL_STORAGE_CAPACITY) ||
            metadata == 0) {
            return false;
        }
        sourceLen = static_cast<uint32_t>(candidateLen);
        firstFill = metadata < 0;
        budget = static_cast<uint32_t>(firstFill ? -metadata : metadata);
        // MTP3 最坏情况下四路 top-2048 完全不重合，因此最低档必须
        // 能容纳 8192 个 token。6144 只属于非 MTP 路径。
        const bool supportedBudget =
            budget == 8192U || budget == 10240U || budget == 12288U;
        if (!supportedBudget || budget > MAX_CACHE_TOKENS ||
            budget > sourceLen) {
            return false;
        }
        const uint32_t earliestVisible =
            static_cast<uint32_t>(finalLen) - (queryCount - 1U);
        return sourceLen <= earliestVisible &&
            sourceLen <= tiling_->maxCandidateLen &&
            finalLen <= static_cast<int32_t>(tiling_->tokenCapacity);
    }

    __aicore__ inline bool ProcessFirstFill(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        uint64_t poolBase, uint32_t sourceLen, uint32_t finalLen,
        uint32_t budget, LocalTensor<int32_t> topk,
        LocalTensor<int32_t> slots, LocalTensor<int32_t> sources,
        LocalTensor<int32_t> cacheChunk,
        LocalTensor<uint8_t> protectedSlots, LocalTensor<uint32_t> hash)
    {
        // First validate and deduplicate the complete MTP union in UB.  Do
        // not clear the caller-owned pool until success is guaranteed: an
        // invalid token or union>C must leave request state byte-for-byte
        // unchanged.
        InitHash(hash);
        uint32_t unionCount = 0;
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            LoadTopk(queryRow, topk);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 ||
                    token >= static_cast<int32_t>(sourceLen)) {
                    return false;
                }
                uint32_t hashPosition = 0;
                bool found = false;
                if (!FindHashPosition(
                        hash, token, hashPosition, found)) {
                    return false;
                }
                if (!found) {
                    if (unionCount >= budget ||
                        unionCount >= tiling_->outputCapacity) {
                        return false;
                    }
                    hash.SetValue(
                        hashPosition,
                        PackHashEntry(
                            static_cast<uint32_t>(token),
                            MISS_SLOT));
                    ++unionCount;
                }
            }
        }

        GlobalTensor<int32_t> poolRow = cacheSlotsPoolGm_[poolBase];
        FillGlobalRange(
            poolRow, tiling_->tokenCapacity, INVALID, cacheChunk);

        uint32_t residentCount = 0;
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            LoadTopk(queryRow, topk);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                uint32_t hashPosition = 0;
                bool found = false;
                if (!FindHashPosition(
                        hash, token, hashPosition, found) || !found) {
                    return false;
                }
                int32_t slot = static_cast<int32_t>(
                    hash.GetValue(hashPosition) >> SLOT_SHIFT);
                if (slot == static_cast<int32_t>(MISS_SLOT)) {
                    slot = static_cast<int32_t>(residentCount);
                    hash.SetValue(
                        hashPosition,
                        PackHashEntry(
                            static_cast<uint32_t>(token),
                            static_cast<uint32_t>(slot)));
                    WritePoolScalar(
                        poolBase + static_cast<uint32_t>(token), slot);
                    WriteCopyPair(batch, residentCount, token, slot);
                    ++residentCount;
                }
            }
        }

        for (uint32_t token = 0;
             token < sourceLen && residentCount < budget; ++token) {
            uint32_t hashPosition = 0;
            bool found = false;
            if (!FindHashPosition(
                    hash, static_cast<int32_t>(token),
                    hashPosition, found)) {
                return false;
            }
            if (found) {
                continue;
            }
            const int32_t slot = static_cast<int32_t>(residentCount);
            hash.SetValue(
                hashPosition,
                PackHashEntry(token, static_cast<uint32_t>(slot)));
            WritePoolScalar(poolBase + token, slot);
            WriteCopyPair(
                batch, residentCount, static_cast<int32_t>(token), slot);
            ++residentCount;
        }
        if (residentCount != budget) {
            return false;
        }
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            if (!PublishSparseSelection(
                    queryRow, queryEnd, sourceLen, finalLen, budget, true,
                    topk, slots, sources, protectedSlots, hash)) {
                return false;
            }
        }
        WritePoolScalar(
            poolBase + tiling_->tokenCapacity,
            static_cast<int32_t>(budget));
        WriteOutputScalar(
            residentSeqLengthsGm_[batch],
            static_cast<int32_t>(budget + TAIL_STORAGE_CAPACITY));
        WriteOutputScalar(copyCountsGm_[batch], static_cast<int32_t>(budget));
        return true;
    }

    __aicore__ inline bool ProcessSteady(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        uint64_t poolBase, uint32_t sourceLen, uint32_t finalLen,
        uint32_t budget, LocalTensor<int32_t> topk,
        LocalTensor<int32_t> slots, LocalTensor<int32_t> sources,
        LocalTensor<int32_t> cacheChunk,
        LocalTensor<uint8_t> protectedSlots,
        LocalTensor<uint32_t> hash)
    {
        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        InitHash(hash);
        uint32_t missCount = 0;
        uint32_t uniqueHitCount = 0;

        // Build one request-local union in UB.  This deliberately avoids
        // reading a pool scalar immediately after writing it through MTE3:
        // those writes are not guaranteed to invalidate the scalar DCache.
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            LoadTopk(queryRow, topk);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 ||
                    token >= static_cast<int32_t>(sourceLen)) {
                    return false;
                }
                uint32_t hashPosition = 0;
                bool found = false;
                if (!FindHashPosition(
                        hash, token, hashPosition, found)) {
                    return false;
                }
                if (found) {
                    continue;
                }
                const uint32_t tokenU32 = static_cast<uint32_t>(token);
                const int32_t slot = cacheSlotsPoolGm_.GetValue(
                    poolBase + tokenU32);
                if (slot >= 0 && slot < static_cast<int32_t>(budget)) {
                    hash.SetValue(
                        hashPosition,
                        PackHashEntry(
                            tokenU32, static_cast<uint32_t>(slot)));
                    protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                    ++uniqueHitCount;
                } else if (slot == INVALID) {
                    if (missCount >= tiling_->outputCapacity) {
                        return false;
                    }
                    hash.SetValue(
                        hashPosition, PackHashEntry(tokenU32, MISS_SLOT));
                    ++missCount;
                } else {
                    return false;
                }
            }
        }
        if (uniqueHitCount + missCount > budget) {
            return false;
        }

        uint32_t assigned = 0;
        uint32_t missHashPosition = 0;
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
            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
            for (uint32_t offset = 0;
                 offset < chunkLen && assigned < missCount; ++offset) {
                const int32_t slot = cacheChunk.GetValue(offset);
                if (slot < 0 || slot >= static_cast<int32_t>(budget) ||
                    protectedSlots.GetValue(static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                while (missHashPosition < HASH_CAPACITY &&
                       (hash.GetValue(missHashPosition) == HASH_EMPTY ||
                        (hash.GetValue(missHashPosition) >> SLOT_SHIFT) !=
                            MISS_SLOT)) {
                    ++missHashPosition;
                }
                if (missHashPosition >= HASH_CAPACITY) {
                    return false;
                }
                const int32_t missToken = static_cast<int32_t>(
                    hash.GetValue(missHashPosition) & TOKEN_MASK);
                const uint32_t victimToken = chunkBase + offset;
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 2);
                hash.SetValue(
                    missHashPosition,
                    PackHashEntry(
                        static_cast<uint32_t>(missToken),
                        static_cast<uint32_t>(slot)));
                WritePoolScalar(poolBase + victimToken, INVALID);
                WritePoolScalar(
                    poolBase + static_cast<uint32_t>(missToken), slot);
                WriteCopyPair(batch, assigned, missToken, slot);
                ++assigned;
                ++missHashPosition;
            }
        }
        if (assigned != missCount) {
            return false;
        }

        // Resolve each query row from the UB union map and publish a paired
        // miss-prefix/hit-suffix source+slot row for fused copy+SFA.
        for (uint32_t queryRow = queryStart; queryRow < queryEnd; ++queryRow) {
            if (!PublishSparseSelection(
                    queryRow, queryEnd, sourceLen, finalLen, budget, false,
                    topk, slots, sources, protectedSlots, hash)) {
                return false;
            }
        }
        WriteOutputScalar(
            residentSeqLengthsGm_[batch],
            static_cast<int32_t>(budget + TAIL_STORAGE_CAPACITY));
        WriteOutputScalar(
            copyCountsGm_[batch], static_cast<int32_t>(missCount));
        return true;
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        uint32_t queryStart = 0;
        uint32_t queryEnd = 0;
        ClearCopyRow(batch);
        if (!QueryRange(batch, queryStart, queryEnd)) {
            WriteOutputScalar(residentSeqLengthsGm_[batch], 0);
            WriteOutputScalar(copyCountsGm_[batch], -1);
            return;
        }

        const int32_t mode = rowModesGm_.GetValue(batch);
        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        if (mode == ROW_MODE_PAD) {
            ProcessPad(batch, queryStart, queryEnd);
            return;
        }
        if (mode == ROW_MODE_DENSE) {
            const int32_t finalLen = finalSeqLengthsKvGm_.GetValue(batch);
            if (!ProcessDense(
                    batch, queryStart, queryEnd, finalLen,
                    topk, slots, sources)) {
                MarkRequestError(batch, queryStart, queryEnd);
            }
            return;
        }
        if (mode != ROW_MODE_SPARSE) {
            MarkRequestError(batch, queryStart, queryEnd);
            return;
        }

        const int32_t poolRow = reqPoolEntriesGm_.GetValue(batch);
        if (poolRow < 0 ||
            poolRow >= static_cast<int32_t>(tiling_->poolSize)) {
            MarkRequestError(batch, queryStart, queryEnd);
            return;
        }
        const uint64_t poolBase =
            static_cast<uint64_t>(poolRow) *
            (static_cast<uint64_t>(tiling_->tokenCapacity) + 1U);
        const int32_t metadata = cacheSlotsPoolGm_.GetValue(
            poolBase + tiling_->tokenCapacity);
        const int32_t candidateLen = candidateLensGm_.GetValue(batch);
        const int32_t finalLen = finalSeqLengthsKvGm_.GetValue(batch);
        const uint32_t queryCount = queryEnd - queryStart;
        uint32_t sourceLen = 0;
        uint32_t budget = 0;
        bool firstFill = false;
        if (!ValidateSparse(
                poolRow, candidateLen, finalLen, metadata, queryCount,
                sourceLen, budget, firstFill)) {
            MarkRequestError(batch, queryStart, queryEnd);
            return;
        }

        const bool ok = firstFill
            ? ProcessFirstFill(
                batch, queryStart, queryEnd, poolBase, sourceLen,
                static_cast<uint32_t>(finalLen), budget, topk, slots,
                sources,
                cacheChunkBuf_.Get<int32_t>(),
                protectedSlotBuf_.Get<uint8_t>(),
                hashBuf_.Get<uint32_t>())
            : ProcessSteady(
                batch, queryStart, queryEnd, poolBase, sourceLen,
                static_cast<uint32_t>(finalLen), budget, topk, slots,
                sources,
                cacheChunkBuf_.Get<int32_t>(),
                protectedSlotBuf_.Get<uint8_t>(),
                hashBuf_.Get<uint32_t>());
        if (!ok) {
            MarkRequestError(batch, queryStart, queryEnd);
        }
    }

private:
    TPipe *pipe_;
    const VllmA5LiManageC8TilingData *tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t topkRowStride_ = TOPK;
    GlobalTensor<int32_t> topkIndicesGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> finalSeqLengthsKvGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsPoolGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> sparseAndTailSlotsGm_;
    GlobalTensor<int32_t> sparseAndTailSrcIdsGm_;
    GlobalTensor<int32_t> perQueryMissCountsGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
    GlobalTensor<int32_t> copySrcIdsGm_;
    GlobalTensor<int32_t> copyDstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TBuf<TPosition::VECCALC> topkBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
    TBuf<TPosition::VECCALC> protectedSlotBuf_;
    TBuf<TPosition::VECCALC> hashBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};
} // namespace vllm_a5_mtp_manager

#endif
