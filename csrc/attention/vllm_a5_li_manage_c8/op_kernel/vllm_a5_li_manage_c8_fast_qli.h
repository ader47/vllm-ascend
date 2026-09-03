/** Stable MTP3 C8 LI stage with {source18, slot14} TopK payloads. */

#ifndef VLLM_A5_LI_MANAGE_C8_FAST_QLI_H
#define VLLM_A5_LI_MANAGE_C8_FAST_QLI_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "vllm_a5_li_manage_c8_tiling.h"
#include "vllm_a5_li_manage_c8_fast_workspace.h"

// The vendored payload implementation has private namespaces and include
// guards, so it can coexist with the stock QuantLI cold path.
using VllmFastLIC8TilingData = VllmA5LiManageC8TilingData;
#define A5_MTP_CLASSIFY_ONLY 1
#define A5_MTP_SLOT_OUTPUT_STRIDE 2176U
#include "arch35_payload_fast_c8/lightning_indexer_common.h"
#include "arch35_payload_fast_c8/lightning_indexer_service_cube.h"
#include "arch35_payload_fast_c8/lightning_indexer_service_vector.h"
#undef A5_MTP_SLOT_OUTPUT_STRIDE
#undef A5_MTP_CLASSIFY_ONLY

namespace vllm_a5_li_manage_c8_fast {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t ROUTES = 4U;
constexpr uint32_t ROW_MODE_SPARSE = 2U;

using C8MtpQliType = VllmFastQLICommon::QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true,
    VllmFastQLICommon::LI_LAYOUT::TND,
    VllmFastQLICommon::LI_LAYOUT::PA_BSND,
    bfloat16_t, float32_t, float32_t, uint16_t>;

/**
 * Runtime gate for the only optimized regime. Any cold/mixed/variable-tail
 * batch returns false and executes the original correctness-first path.
 */
__aicore__ inline bool IsAllStableMtp3(
    GM_ADDR actualSeqLengthsQuery, GM_ADDR candidateLens,
    GM_ADDR finalSeqLengthsKv, GM_ADDR rowModes, GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool, const VllmA5LiManageC8TilingData *tiling)
{
    if (tiling->fastPathEnabled == 0U ||
        tiling->totalQueryRows != tiling->batchSize * ROUTES) {
        return false;
    }
    GlobalTensor<int32_t> queryEnds;
    GlobalTensor<int32_t> candidates;
    GlobalTensor<int32_t> finalLengths;
    GlobalTensor<int32_t> modes;
    GlobalTensor<int32_t> entries;
    GlobalTensor<int32_t> pool;
    queryEnds.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
    candidates.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
    finalLengths.SetGlobalBuffer((__gm__ int32_t *)finalSeqLengthsKv);
    modes.SetGlobalBuffer((__gm__ int32_t *)rowModes);
    entries.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
    pool.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);

    const uint64_t poolStride =
        static_cast<uint64_t>(tiling->tokenCapacity) + 1U;
    for (uint32_t batch = 0U; batch < tiling->batchSize; ++batch) {
        if (queryEnds.GetValue(batch) !=
                static_cast<int32_t>((batch + 1U) * ROUTES) ||
            modes.GetValue(batch) != static_cast<int32_t>(ROW_MODE_SPARSE)) {
            return false;
        }
        const int32_t poolRow = entries.GetValue(batch);
        const int32_t candidate = candidates.GetValue(batch);
        const int32_t finalLen = finalLengths.GetValue(batch);
        if (poolRow < 0 ||
            poolRow >= static_cast<int32_t>(tiling->poolSize) ||
            candidate <= static_cast<int32_t>(TOPK) ||
            candidate > static_cast<int32_t>(tiling->tokenCapacity) ||
            candidate > static_cast<int32_t>(tiling->maxCandidateLen) ||
            candidate % static_cast<int32_t>(BLOCK_SIZE) != 0 ||
            finalLen < candidate ||
            finalLen > static_cast<int32_t>(tiling->tokenCapacity) ||
            finalLen - candidate > 256) {
            return false;
        }
        const int32_t budget = pool.GetValue(
            static_cast<uint64_t>(poolRow) * poolStride +
            tiling->tokenCapacity);
        if ((budget != 8192 && budget != 10240 && budget != 12288) ||
            budget > candidate) {
            return false;
        }
        // The payload implementation scores one shared source range for all
        // four speculative rows. Boundary steps whose causal full-block
        // lengths differ remain on the old path.
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            const uint32_t later = ROUTES - 1U - route;
            if (finalLen <= static_cast<int32_t>(later)) {
                return false;
            }
            const uint32_t visible =
                static_cast<uint32_t>(finalLen) - later;
            const uint32_t selection =
                ((visible - 1U) / BLOCK_SIZE) * BLOCK_SIZE;
            if (selection != static_cast<uint32_t>(candidate)) {
                return false;
            }
        }
    }
    return true;
}

class QuantLiMtpPhase {
public:
    __aicore__ inline QuantLiMtpPhase(
        TPipe *pipe, const VllmA5LiManageC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
        GM_ADDR indexKeyCache, GM_ADDR indexKeyDequantScale,
        GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
        GM_ADDR candidateLens, GM_ADDR indexBlockTable,
        GM_ADDR routePairRows, GM_ADDR topkSlots,
        GM_ADDR topkSourceIds,
        GM_ADDR routeThresholds, GM_ADDR routeMissCounts,
        GM_ADDR userWorkspace)
    {
        subBlockIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            aiCoreIdx_ = subBlockIdx_;
        }

        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)indexBlockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.tSize = tiling_->totalQueryRows;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = HEAD_DIM;
        constInfo_.sparseCount = TOPK;
        constInfo_.kSeqSize = tiling_->fastScoreRowStride;
        constInfo_.qSeqSize = ROUTES;
        constInfo_.kCacheBlockSize = BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = VllmFastQLICommon::LI_LAYOUT::TND;
        // candidate_lens is already the durable, fully offloaded prefix.
        // It excludes the causal tail/current queries, so every MTP route
        // ranks the complete prefix; causal handling is applied only while
        // publishing each route's dense tail slots.
        constInfo_.attenMaskFlag = false;
        constInfo_.isAccumSeqS1 = true;
        constInfo_.s1BaseSize = tiling_->fastQueryTileSize;
        constInfo_.mBaseSize =
            constInfo_.s1BaseSize * tiling_->indexHeads;
        constInfo_.s2BaseSize = BLOCK_SIZE;
        constInfo_.keyStride0 = tiling_->keyStride;
        constInfo_.keyDequantScaleStride0 = tiling_->scaleStride;
        constInfo_.poolSize = tiling_->poolSize;
        // The last int32 is the persistent +/-C state, not a token mapping.
        constInfo_.cacheSlotsSize =
            static_cast<uint64_t>(tiling_->tokenCapacity) + 1U;
        constInfo_.setL2DisableFlag = false;

        scoreWorkspaceBaseGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)userWorkspace);
        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)indexWeights);
            queryScaleGm_.SetGlobalBuffer(
                (__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer(
                (__gm__ float *)indexKeyDequantScale);
            routePairRowsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)routePairRows);
            topkSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
            topkSourceIdsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)topkSourceIds);
            routeMissCountsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)routeMissCounts);
            routeThresholdsGm_.SetGlobalBuffer(
                (__gm__ uint16_t *)routeThresholds);
            vectorService_.InitParams(constInfo_, tiling_);
            vectorService_.InitVecInputTensor(
                weightsGm_, queryScaleGm_, keyScaleGm_,
                routePairRowsGm_, blockTableGm_, cacheSlotsGm_,
                topkSlotsGm_, routeMissCountsGm_);
            // Stage 1 publishes the complete miss-prefix/hit-suffix source
            // row and the hit-slot suffix. Stage 2 repairs only miss slots and
            // appends the paired causal tail.
            vectorService_.InitMtpTopkSourceTensor(topkSourceIdsGm_);
            vectorService_.InitMtpThresholdTensor(routeThresholdsGm_);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer(
                (__gm__ fp8_e4m3fn_t *)indexKeyCache);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(
                blockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t tile = tiling_->fastQueryTileSize;
        if (tile != 2U && tile != ROUTES) {
            return;
        }
        const uint32_t queryTileCount = ROUTES / tile;
        const uint32_t taskCount = tiling_->batchSize * queryTileCount;
        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4,
                             PIPE_V>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4,
                             PIPE_V>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0U;
        for (uint32_t task = aiCoreIdx_; task < taskCount;
             task += tiling_->usedCoreNum) {
            const uint32_t batch = task / queryTileCount;
            const uint32_t gS1 = task % queryTileCount;
            const uint32_t queryBegin = batch * ROUTES;
            const uint32_t candidate = static_cast<uint32_t>(
                candidateLensGm_.GetValue(batch));
            const uint32_t poolRow = static_cast<uint32_t>(
                reqPoolEntriesGm_.GetValue(batch));
            const uint64_t poolStride =
                static_cast<uint64_t>(tiling_->tokenCapacity) + 1U;
            const uint32_t budget = static_cast<uint32_t>(
                cacheSlotsGm_.GetValue(
                    static_cast<uint64_t>(poolRow) * poolStride +
                    tiling_->tokenCapacity));
            const uint32_t loopCount = candidate / BLOCK_SIZE;
            if ASCEND_IS_AIV {
                const uint64_t requestScoreBase =
                    static_cast<uint64_t>(batch) *
                    (tiling_->fastScoreWorkspaceStride / sizeof(uint16_t));
                const uint64_t routeScoreBase =
                    static_cast<uint64_t>(gS1) * tile *
                    tiling_->fastScoreRowStride;
                vectorService_.InitVecWorkspaceTensor(
                    scoreWorkspaceBaseGm_[
                        requestScoreBase + routeScoreBase]);
            }
            for (uint32_t s2 = 0U; s2 < loopCount;
                 ++s2, ++globalLoop) {
                VllmFastQLICommon::RunInfo run{};
                run.loop = globalLoop;
                run.bN2Idx = batch;
                run.bIdx = batch;
                run.n2Idx = 0U;
                run.gS1Idx = gS1;
                run.s2Idx = s2;
                run.s2Start = 0U;
                run.validS2Len = candidate;
                run.kScaleLoop = s2 / 16U + 1U;
                run.cacheRowIdx = poolRow;
                run.cacheTokenCount = budget;
                run.actS1Size = ROUTES;
                run.actS2Size = candidate;
                run.actS2SizeOrig = candidate;
                run.actMBaseSize = constInfo_.mBaseSize;
                run.actualSingleProcessSInnerSize = BLOCK_SIZE;
                run.actualSingleProcessSInnerSizeAlign = BLOCK_SIZE;
                run.tensorQueryOffset =
                    static_cast<uint64_t>(queryBegin) *
                        tiling_->indexHeads * HEAD_DIM +
                    static_cast<uint64_t>(gS1) *
                        constInfo_.mBaseSize * HEAD_DIM;
                run.tensorKeyOffset =
                    static_cast<uint64_t>(s2) * BLOCK_SIZE * HEAD_DIM;
                run.tensorKeyScaleOffset =
                    static_cast<uint64_t>(s2) * BLOCK_SIZE;
                run.tensorWeightsOffset =
                    static_cast<uint64_t>(queryBegin) *
                    tiling_->weightStride;
                run.tensorQueryScaleOffset =
                    static_cast<uint64_t>(queryBegin) *
                    tiling_->indexHeads;
                run.indiceOutOffset =
                    static_cast<uint64_t>(queryBegin) * TOPK;
                run.isFirstS2InnerLoop = s2 == 0U;
                run.isLastS2InnerLoop = s2 + 1U == loopCount;
                run.isAllLoopEnd = false;
                run.isValid = true;

                if ASCEND_IS_AIC {
                    matmulService_.ComputeMm1(run);
                } else {
                    vectorService_.ProcessVec1(run);
                    if (run.isLastS2InnerLoop) {
                        const bool finalTask =
                            task + tiling_->usedCoreNum >= taskCount;
                        vectorService_.ProcessTopK(run, finalTask);
                    }
                }
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<
                VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<
                VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    TPipe *pipe_;
    const VllmA5LiManageC8TilingData *tiling_;
    uint32_t subBlockIdx_ = 0U;
    uint32_t aiCoreIdx_ = 0U;
    VllmFastQLICommon::ConstInfo constInfo_{};
    VllmFastQLIKernel::QLIMatmul<C8MtpQliType> matmulService_;
    VllmFastQLIKernel::QLIVector<C8MtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> routePairRowsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> topkSourceIdsGm_;
    GlobalTensor<int32_t> routeMissCountsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceBaseGm_;
};

} // namespace vllm_a5_li_manage_c8_fast

#endif
