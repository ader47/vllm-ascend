/** Stage 2-4: ordered union, victim selection, cache update and slot repair. */

#ifndef VLLM_A5_LI_MANAGE_C8_FAST_UNION_H
#define VLLM_A5_LI_MANAGE_C8_FAST_UNION_H

#include "kernel_operator.h"
#include "vllm_a5_li_manage_c8_fast_victim_vf.h"
#include "vllm_a5_li_manage_c8_fast_workspace.h"

namespace vllm_a5_li_manage_c8_fast {
using namespace AscendC;

constexpr uint32_t UNION_ROUTES = 4U;
constexpr uint32_t UNION_TOPK = 2048U;
constexpr uint32_t UNION_CAPACITY = UNION_ROUTES * UNION_TOPK;
constexpr uint32_t UNION_PAIR_WORDS = UNION_TOPK * 2U;
constexpr uint32_t UNION_SOURCE_MASK = (1U << 18U) - 1U;
constexpr uint32_t UNION_KEY_BASE_BITS = 0x40000000U;
constexpr int32_t UNION_KEY_DECODE_BASE =
    static_cast<int32_t>(UNION_KEY_BASE_BITS + UNION_SOURCE_MASK);
constexpr uint32_t B32_VECTOR_ELEMENTS = 64U;
constexpr uint32_t B32_VECTOR_REPEAT_STRIDE = 8U;
constexpr uint32_t VICTIM_SCAN_CHUNK = 2048U;
constexpr uint32_t VICTIM_SLOT_SHIFT = 18U;
constexpr uint32_t VICTIM_SLOT_MASK = (1U << 14U) - 1U;

template <HardEvent event>
__aicore__ inline void UnionSync(HardEvent value)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(value));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

__aicore__ inline void ExtractPairKeys(
    const LocalTensor<uint32_t> &keys,
    const LocalTensor<uint32_t> &pairs,
    uint32_t count)
{
    GatherMaskParams params;
    params.repeatTimes =
        (count * 2U * sizeof(uint32_t) + 255U) / 256U;
    params.src0BlockStride = 1;
    params.src0RepeatStride = B32_VECTOR_REPEAT_STRIDE;
    params.src1RepeatStride = 0;
    uint64_t reserved = 0U;
    GatherMask(keys, pairs, static_cast<uint8_t>(1), false,
               static_cast<uint32_t>(0), params, reserved);
    PipeBarrier<PIPE_V>();
}

class OrderedMissUnion {
public:
    __aicore__ inline void Init(
        GM_ADDR routePairs, GM_ADDR routeThresholds, GM_ADDR routeCounts,
        GM_ADDR scoreWorkspace, GM_ADDR candidateLens,
        GM_ADDR finalSeqLengthsKv, GM_ADDR reqPoolEntries, GM_ADDR cacheSlots,
        GM_ADDR unionSources, GM_ADDR unionDestinations,
        GM_ADDR unionCounts, GM_ADDR topkSlots,
        GM_ADDR sparseAndTailSlots, GM_ADDR residentSeqLengths,
        uint32_t tokenCapacity, uint32_t outputCapacity,
        uint32_t batchSize, TPipe *pipe)
    {
        routePairsGm_.SetGlobalBuffer((__gm__ int32_t *)routePairs);
        routeThresholdsGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)routeThresholds);
        routeCountsGm_.SetGlobalBuffer((__gm__ int32_t *)routeCounts);
        scoreWorkspaceGm_.SetGlobalBuffer((__gm__ uint16_t *)scoreWorkspace);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        finalSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)finalSeqLengthsKv);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        unionSourcesGm_.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        unionDestinationsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)unionDestinations);
        unionCountsGm_.SetGlobalBuffer((__gm__ int32_t *)unionCounts);
        topkSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        sparseAndTailSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sparseAndTailSlots);
        residentSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentSeqLengths);
        sourceCapacity_ = tokenCapacity;
        poolStride_ = tokenCapacity + 1U;
        outputCapacity_ = outputCapacity;
        batchSize_ = batchSize;
        pipe->InitBuffer(pairInputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(sourceBuf_,
                         UNION_CAPACITY * sizeof(int32_t));
        pipe->InitBuffer(countBuf_, 32U);
        pipe->InitBuffer(
            thresholdBuf_,
            UNION_ROUTES *
                vllm_a5_li_manage_c8_fast_workspace::THRESHOLD_STRIDE *
                sizeof(uint16_t));
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize_; batch += stride) {
            ProcessRequest(batch);
        }
    }

private:
    __aicore__ inline uint32_t ScalarDeduplicate(
        LocalTensor<uint32_t> keys, uint32_t total,
        LocalTensor<int32_t> output)
    {
        uint32_t count = 0U;
        int32_t last = -1;
        for (uint32_t index = 0U; index < total; ++index) {
            const uint32_t key = keys.GetValue(index);
            const int32_t source = static_cast<int32_t>(
                UNION_SOURCE_MASK - (key - UNION_KEY_BASE_BITS));
            if (source != last) {
                output.SetValue(count++, source);
                last = source;
            }
        }
        return count;
    }

    __aicore__ inline uint32_t Deduplicate(
        LocalTensor<float> merged, LocalTensor<float> scratch,
        uint32_t total, LocalTensor<int32_t> output)
    {
        LocalTensor<uint32_t> keys = scratch.ReinterpretCast<uint32_t>();
        ExtractPairKeys(keys, merged.ReinterpretCast<uint32_t>(), total);
        // MrgSort has already put equal source keys next to one another.
        // Keep this boundary correctness-first: GatherMask's predicate
        // repeat geometry is easy to get wrong when the sum of four route
        // miss counts is not a 64-element multiple, and that used to leak a
        // duplicate source into copy_count. Stable decode normally has only
        // a few hundred route misses, so this bounded UB walk is cheap; the
        // scoring/TopK stages remain fully vectorized.
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        return ScalarDeduplicate(keys, total, output);
    }

    __aicore__ inline uint32_t HashVictimScanSeed(
        uint32_t candidate, uint32_t poolRow)
    {
        uint32_t value = candidate ^ ((poolRow + 1U) * 0x9e3779b9U);
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

    __aicore__ inline uint32_t ReadBudget(
        uint32_t batch, uint32_t poolRow)
    {
        (void)batch;
        return static_cast<uint32_t>(cacheSlotsGm_.GetValue(
            static_cast<uint64_t>(poolRow) * poolStride_ +
            sourceCapacity_));
    }

    __aicore__ inline void LoadThresholds(
        uint32_t batch, LocalTensor<uint16_t> local,
        uint16_t values[UNION_ROUTES])
    {
        constexpr uint32_t STRIDE =
            vllm_a5_li_manage_c8_fast_workspace::THRESHOLD_STRIDE;
        const uint64_t routeBase =
            static_cast<uint64_t>(batch) * UNION_ROUTES;
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            DataCopyPad(
                local[route * STRIDE],
                routeThresholdsGm_[(routeBase + route) * STRIDE],
                {1, STRIDE * static_cast<uint32_t>(sizeof(uint16_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
        }
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            values[route] = local.GetValue(route * STRIDE);
        }
    }

    __aicore__ inline uint32_t CompactSafeVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t required,
        const uint16_t thresholds[UNION_ROUTES],
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        LocalTensor<uint8_t> scratch = pairInputBuf_.Get<uint8_t>();
        LocalTensor<uint16_t> score0 =
            scratch.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score1 = score0[VICTIM_SCAN_CHUNK];
        LocalTensor<uint16_t> score2 =
            score0[VICTIM_SCAN_CHUNK * 2U];
        LocalTensor<uint16_t> score3 =
            score0[VICTIM_SCAN_CHUNK * 3U];
        LocalTensor<int32_t> slots32 =
            scratch[VICTIM_SCAN_CHUNK * UNION_ROUTES *
                    sizeof(uint16_t)].ReinterpretCast<int32_t>();
        LocalTensor<int16_t> slots16 =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t))]
                .ReinterpretCast<int16_t>();
        LocalTensor<uint32_t> compact =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t) +
                     sizeof(int16_t))].ReinterpretCast<uint32_t>();

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk =
            HashVictimScanSeed(candidate, poolRow) % chunks;
        const uint64_t requestScoreBase =
            static_cast<uint64_t>(batch) * UNION_ROUTES * sourceCapacity_;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        uint32_t written = 0U;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t chunkBase = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t chunkLen =
                chunkBase + VICTIM_SCAN_CHUNK > candidate
                    ? candidate - chunkBase
                    : VICTIM_SCAN_CHUNK;
            const uint32_t alignedLen =
                (chunkLen + B32_VECTOR_ELEMENTS - 1U) /
                B32_VECTOR_ELEMENTS * B32_VECTOR_ELEMENTS;
            const DataCopyExtParams scoreCopy{
                1, chunkLen * static_cast<uint32_t>(sizeof(uint16_t)),
                0, 0, 0};
            const DataCopyPadExtParams<uint16_t> scorePad{
                true, 0,
                static_cast<uint8_t>(alignedLen - chunkLen), 0U};
            DataCopyPad(
                score0,
                scoreWorkspaceGm_[requestScoreBase + chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                score1,
                scoreWorkspaceGm_[requestScoreBase + sourceCapacity_ +
                                  chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                score2,
                scoreWorkspaceGm_[requestScoreBase +
                                  sourceCapacity_ * 2U + chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                score3,
                scoreWorkspaceGm_[requestScoreBase +
                                  sourceCapacity_ * 3U + chunkBase],
                scoreCopy, scorePad);
            DataCopyPad(
                slots32, cacheSlotsGm_[cacheBase + chunkBase],
                {1, chunkLen * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            Cast(slots16, slots32, RoundMode::CAST_NONE, chunkLen);
            if (alignedLen > chunkLen) {
                Duplicate(
                    slots16[chunkLen], static_cast<int16_t>(-1),
                    alignedLen - chunkLen);
            }
            PipeBarrier<PIPE_V>();
            VllmA5LiManageC8FastVictimVF::CompactEligiblePayloads(
                (__ubuf__ uint32_t *)compact.GetPhyAddr(),
                (__ubuf__ uint16_t *)score0.GetPhyAddr(),
                (__ubuf__ uint16_t *)score1.GetPhyAddr(),
                (__ubuf__ uint16_t *)score2.GetPhyAddr(),
                (__ubuf__ uint16_t *)score3.GetPhyAddr(),
                (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
                thresholds[0], thresholds[1], thresholds[2],
                thresholds[3], static_cast<uint16_t>(budget),
                chunkBase, alignedLen / B32_VECTOR_ELEMENTS);
            const uint32_t compactCount = static_cast<uint32_t>(
                GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
            PipeBarrier<PIPE_V>();
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
            const uint32_t remaining = required - written;
            const uint32_t keep =
                compactCount < remaining ? compactCount : remaining;
            for (uint32_t index = 0U; index < keep; ++index) {
                const uint32_t payload = compact.GetValue(index);
                destinations.SetValue(
                    written + index,
                    static_cast<int32_t>(
                        (payload >> VICTIM_SLOT_SHIFT) & VICTIM_SLOT_MASK));
                victimSources.SetValue(
                    written + index,
                    static_cast<int32_t>(payload & UNION_SOURCE_MASK));
            }
            written += keep;
        }
        return written;
    }

    __aicore__ inline uint32_t AppendExactVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t written, uint32_t required,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (written >= required) {
            return written;
        }
        // One int32 marker per physical cache slot fits in pairInputBuf_
        // because the MTP cache budget is capped at 12288.
        LocalTensor<int32_t> protectedSlots =
            pairInputBuf_.Get<int32_t>();
        Duplicate(protectedSlots, static_cast<int32_t>(0), budget);
        PipeBarrier<PIPE_V>();
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t index = 0U; index < written; ++index) {
            const int32_t slot = destinations.GetValue(index);
            if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
            }
        }
        // Stage 1 writes directly to the public [T, 2176] rows. Only the
        // first 2048 entries participate in sparse TopK protection; the tail
        // suffix is filled after cache management completes.
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            const uint64_t topkBase =
                (static_cast<uint64_t>(batch) * UNION_ROUTES + route) *
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY;
            for (uint32_t index = 0U; index < UNION_TOPK; ++index) {
                const int32_t slot =
                    topkSlotsGm_.GetValue(topkBase + index);
                if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                    protectedSlots.SetValue(
                        static_cast<uint32_t>(slot), 1);
                }
            }
        }

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk =
            HashVictimScanSeed(candidate, poolRow) % chunks;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t begin = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t end =
                begin + VICTIM_SCAN_CHUNK < candidate
                    ? begin + VICTIM_SCAN_CHUNK
                    : candidate;
            for (uint32_t source = begin;
                 source < end && written < required; ++source) {
                const int32_t slot =
                    cacheSlotsGm_.GetValue(cacheBase + source);
                if (slot < 0 || static_cast<uint32_t>(slot) >= budget ||
                    protectedSlots.GetValue(
                        static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                destinations.SetValue(written, slot);
                victimSources.SetValue(
                    written, static_cast<int32_t>(source));
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++written;
            }
        }
        return written;
    }

    __aicore__ inline uint32_t FindVictims(
        uint32_t batch, uint32_t count,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (count == 0U) {
            return 0U;
        }
        Duplicate(destinations, static_cast<int32_t>(-1), count);
        Duplicate(victimSources, static_cast<int32_t>(-1), count);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> thresholdLocal = thresholdBuf_.Get<uint16_t>();
        uint16_t thresholds[UNION_ROUTES];
        LoadThresholds(batch, thresholdLocal, thresholds);
        const uint32_t candidate = static_cast<uint32_t>(
            candidateLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
        uint32_t written = CompactSafeVictims(
            batch, candidate, poolRow, budget, count, thresholds,
            destinations, victimSources);
        written = AppendExactVictims(
            batch, candidate, poolRow, budget, written, count,
            destinations, victimSources);
        return written;
    }

    __aicore__ inline uint32_t ApplyCacheUpdates(
        uint32_t batch, uint32_t count,
        LocalTensor<int32_t> sources,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        const uint32_t candidate = static_cast<uint32_t>(
            candidateLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        uint32_t updated = 0U;
        for (uint32_t index = 0U; index < count; ++index) {
            const int32_t source = sources.GetValue(index);
            const int32_t destination = destinations.GetValue(index);
            const int32_t victimSource = victimSources.GetValue(index);
            if (source < 0 || victimSource < 0 || destination < 0 ||
                static_cast<uint32_t>(source) >= candidate ||
                static_cast<uint32_t>(victimSource) >= candidate ||
                static_cast<uint32_t>(destination) >= budget) {
                break;
            }
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(victimSource), -1);
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(source), destination);
            ++updated;
        }
        if (updated != 0U) {
            PipeBarrier<PIPE_ALL>();
        }
        return updated;
    }

    __aicore__ inline void PrepareTopkMissPrefixes(
        uint32_t batch, const uint32_t lengths[UNION_ROUTES],
        uint32_t unionCount, LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> allDestinations)
    {
        LocalTensor<int32_t> routePairs = pairInputBuf_.Get<int32_t>();
        // pairInputBuf_ has 16384 int32 words. One 4096-word interleaved
        // route-pair row plus four 2048-word destination prefixes use only
        // 12288 words and remain disjoint.
        const uint64_t requestPairBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            const uint32_t length = lengths[route];
            if (length == 0U) {
                continue;
            }
            const uint32_t pairOffset = route * UNION_PAIR_WORDS;
            DataCopyPad(
                routePairs,
                routePairsGm_[requestPairBase + pairOffset],
                {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            LocalTensor<int32_t> rowDestinations =
                allDestinations[route * UNION_TOPK];
            uint32_t unionCursor = 0U;
            for (uint32_t miss = 0U; miss < length; ++miss) {
                const int32_t source = routePairs.GetValue(miss * 2U + 1U);
                while (unionCursor < unionCount &&
                       unionSources.GetValue(unionCursor) < source) {
                    ++unionCursor;
                }
                const int32_t destination =
                    unionCursor < unionCount &&
                            unionSources.GetValue(unionCursor) == source
                        ? unionDestinations.GetValue(unionCursor)
                        : -1;
                rowDestinations.SetValue(miss, destination);
            }
        }
    }

    __aicore__ inline void PublishFinalOutputs(
        uint32_t batch, const uint32_t lengths[UNION_ROUTES],
        uint32_t count, LocalTensor<int32_t> countLocal,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> topkMissDestinations)
    {
        countLocal.SetValue(0U, static_cast<int32_t>(count));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (count != 0U) {
            const uint64_t unionOffset =
                static_cast<uint64_t>(batch) * outputCapacity_;
            const uint16_t unionBytes = static_cast<uint16_t>(
                count * static_cast<uint32_t>(sizeof(int32_t)));
            DataCopyPad(
                unionSourcesGm_[unionOffset], unionSources,
                {1, unionBytes, 0, 0});
            DataCopyPad(
                unionDestinationsGm_[unionOffset], unionDestinations,
                {1, unionBytes, 0, 0});
            for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                const uint64_t rowOffset =
                    (static_cast<uint64_t>(batch) * UNION_ROUTES + route) *
                    vllm_a5_li_manage_c8_fast_workspace::
                        ATTENTION_CAPACITY;
                DataCopyPad(
                    topkSlotsGm_[rowOffset],
                    topkMissDestinations[route * UNION_TOPK],
                    {1, static_cast<uint16_t>(
                            lengths[route] * sizeof(int32_t)),
                     0, 0});
            }
        }
        DataCopyPad(
            unionCountsGm_[batch], countLocal,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        // Stage 1 already published each 2048-slot sparse prefix directly to
        // the caller-owned [T,2176] output, and the loop above repaired only
        // its miss prefix. Publish the 128-entry causal-tail suffix in place;
        // this removes the old full-row GM -> UB -> GM round trip.
        LocalTensor<int32_t> tail = pairInputBuf_.Get<int32_t>();
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
        const uint32_t finalLen = static_cast<uint32_t>(
            finalSeqLengthsKvGm_.GetValue(batch));
        const uint32_t candidate = static_cast<uint32_t>(
            candidateLensGm_.GetValue(batch));
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            Duplicate(tail, static_cast<int32_t>(-1),
                      vllm_a5_li_manage_c8_fast_workspace::
                          ATTENTION_CAPACITY - UNION_TOPK);
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
            const uint32_t later = UNION_ROUTES - 1U - route;
            const uint32_t visible = finalLen - later;
            const uint32_t tailCount = visible - candidate;
            for (uint32_t index = 0U; index < tailCount; ++index) {
                const uint32_t token = candidate + index;
                tail.SetValue(
                    index,
                    static_cast<int32_t>(budget + token % 256U));
            }
            UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            const uint64_t publicRow =
                (static_cast<uint64_t>(batch) * UNION_ROUTES + route) *
                vllm_a5_li_manage_c8_fast_workspace::ATTENTION_CAPACITY;
            DataCopy(
                sparseAndTailSlotsGm_[publicRow + UNION_TOPK], tail,
                vllm_a5_li_manage_c8_fast_workspace::
                    ATTENTION_CAPACITY - UNION_TOPK);
            UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        countLocal.SetValue(
            0U, static_cast<int32_t>(budget + 2U * 128U));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(
            residentSeqLengthsGm_[batch], countLocal,
            {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        LocalTensor<float> pairs = pairInputBuf_.Get<float>();
        LocalTensor<float> merged = pairOutputBuf_.Get<float>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> countLocal = countBuf_.Get<int32_t>();
        const uint64_t routeBase =
            static_cast<uint64_t>(batch) * UNION_CAPACITY * 2U;
        DataCopyPad(countLocal, routeCountsGm_[batch * UNION_ROUTES],
                    {1, static_cast<uint32_t>(UNION_ROUTES * sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[UNION_ROUTES] = {0U, 0U, 0U, 0U};
        uint32_t total = 0U;
        LocalTensor<int32_t> pairWords = pairs.ReinterpretCast<int32_t>();
        for (uint32_t route = 0U; route < UNION_ROUTES; ++route) {
            int32_t rawLength = countLocal.GetValue(route);
            uint32_t length = rawLength < 0
                ? 0U : static_cast<uint32_t>(rawLength);
            if (length > UNION_TOPK) {
                length = UNION_TOPK;
            }
            lengths[route] = length;
            total += length;
            if (length > 0U) {
                const uint32_t pairOffset = route * UNION_PAIR_WORDS;
                DataCopyPad(
                    pairWords[pairOffset],
                    routePairsGm_[routeBase + pairOffset],
                    {1, static_cast<uint32_t>(
                            length * 2U * sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        if (total == 0U) {
            PublishFinalOutputs(
                batch, lengths, 0U, countLocal, sources, sources, sources);
            return;
        }

        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit =
            (lengths[0] > 0U ? 0b0001 : 0U) |
            (lengths[1] > 0U ? 0b0010 : 0U) |
            (lengths[2] > 0U ? 0b0100 : 0U) |
            (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> inputs;
        inputs.src1 = pairs;
        inputs.src2 = pairs[UNION_PAIR_WORDS];
        inputs.src3 = pairs[UNION_PAIR_WORDS * 2U];
        inputs.src4 = pairs[UNION_PAIR_WORDS * 3U];
        MrgSort<float>(merged, inputs, params);
        PipeBarrier<PIPE_V>();

        const uint32_t count = Deduplicate(merged, pairs, total, sources);
        LocalTensor<int32_t> victimStorage =
            pairOutputBuf_.Get<int32_t>();
        LocalTensor<int32_t> destinations = victimStorage;
        LocalTensor<int32_t> victimSources =
            victimStorage[UNION_CAPACITY];
        const uint32_t found = FindVictims(
            batch, count, destinations, victimSources);
        const uint32_t updated = found == count
            ? ApplyCacheUpdates(
                  batch, count, sources, destinations, victimSources)
            : 0U;
        LocalTensor<int32_t> topkScratch = pairInputBuf_.Get<int32_t>();
        LocalTensor<int32_t> topkMissDestinations =
            topkScratch[UNION_PAIR_WORDS];
        PrepareTopkMissPrefixes(
            batch, lengths, updated, sources, destinations,
            topkMissDestinations);
        PublishFinalOutputs(
            batch, lengths, updated, countLocal, sources, destinations,
            topkMissDestinations);
    }

    GlobalTensor<int32_t> routePairsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<int32_t> routeCountsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> finalSeqLengthsKvGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> unionSourcesGm_;
    GlobalTensor<int32_t> unionDestinationsGm_;
    GlobalTensor<int32_t> unionCountsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> sparseAndTailSlotsGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
    TBuf<TPosition::VECCALC> pairInputBuf_;
    TBuf<TPosition::VECCALC> pairOutputBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> thresholdBuf_;
    uint32_t sourceCapacity_ = 0U;
    uint32_t poolStride_ = 0U;
    uint32_t outputCapacity_ = 0U;
    uint32_t batchSize_ = 0U;
};
} // namespace vllm_a5_li_manage_c8_fast

#endif
