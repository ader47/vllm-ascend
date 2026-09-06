/**
 * A5 MTP0..3 C8 LightningIndexer + request-pool management.
 *
 * All legal batches without first-fill use the unified QLI/union path.
 * First-fill retains the correctness-first manager. Both paths score only
 * the durable SPARSE prefix and append up to 256 causal parity-tail tokens.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "vllm_a5_li_manage_c8_tiling.h"
#include "arch35/quant_lightning_indexer_common.h"
#include "arch35/quant_lightning_indexer_service_cube.h"
#include "arch35/quant_lightning_indexer_service_vector.h"
#include "vllm_a5_li_manage_c8_manager.h"
#include "vllm_a5_li_manage_c8_fast_qli.h"
#include "vllm_a5_li_manage_c8_fast_union.h"
#include "vllm_a5_li_manage_c8_fast_workspace.h"

namespace {
using namespace AscendC;
using namespace QLICommon;
using namespace QLIKernel;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t TOPK = 2048;
constexpr uint32_t ATTENTION_CAPACITY = TOPK + 2U * BLOCK_SIZE;
constexpr uint32_t MAX_QUERIES_PER_REQUEST = 4;
constexpr int32_t ROW_MODE_DENSE = 1;
constexpr int32_t ROW_MODE_SPARSE = 2;
constexpr uint32_t REQUEST_DONE_EVENT = 6;
constexpr uint32_t GATE_READY_EVENT = 4;
constexpr uint32_t GATE_ACK_EVENT = 5;

// Every AIC/AIV must read the same pre-launch +/-C metadata. In particular,
// an early first-fill manager must not change it before a late core decides
// which branch to execute. Drain both local handshakes around the AIV-wide
// barrier before either service allocates its own cross-core events.
__aicore__ inline void SynchronizePathDecision()
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(GATE_READY_EVENT);
        CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
            GATE_READY_EVENT + ConstInfo::AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(GATE_ACK_EVENT);
        CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
            GATE_ACK_EVENT + ConstInfo::AIV0_AIV1_OFFSET);
    } else {
        CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(GATE_READY_EVENT);
        AscendC::SyncAll();
        CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(GATE_ACK_EVENT);
    }
}

using MtpQliType = QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, true,
    LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>;

class VllmA5MtpC8QliPhase {
public:
    __aicore__ inline VllmA5MtpC8QliPhase(
        TPipe *pipe, const VllmA5LiManageC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
        GM_ADDR actualSeqLengthsQuery, GM_ADDR indexKeyCache,
        GM_ADDR indexKeyDequantScale, GM_ADDR indexBlockTable,
        GM_ADDR candidateLens, GM_ADDR finalSeqLengthsKv,
        GM_ADDR rowModes,
        GM_ADDR sparseAndTailSlots, GM_ADDR userWorkspace)
    {
        if ASCEND_IS_AIV {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_;
        }

        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        finalSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)finalSeqLengthsKv);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        indexBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t *)indexBlockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = HEAD_DIM;
        constInfo_.sparseCount = TOPK;
        constInfo_.kSeqSize = tiling_->maxCandidateLen;
        constInfo_.qSeqSize = 1;
        constInfo_.kCacheBlockSize = BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = LI_LAYOUT::TND;
        constInfo_.attenMaskFlag = false;
        constInfo_.cmpRatio = 1;
        constInfo_.batchSupperFlag = false;
        constInfo_.stride = tiling_->keyStride;
        constInfo_.scaleStride = tiling_->scaleStride;
        constInfo_.mBaseSize = 256;
        constInfo_.s1BaseSize =
            (constInfo_.mBaseSize + constInfo_.gSize - 1U) /
            constInfo_.gSize;
        constInfo_.s2BaseSize = BLOCK_SIZE;

        GlobalTensor<uint16_t> scoreWorkspace;
        scoreWorkspace.SetGlobalBuffer(
            (__gm__ uint16_t *)(userWorkspace +
                static_cast<uint64_t>(aiCoreIdx_) *
                    tiling_->scoreWorkspaceStride));

        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)indexWeights);
            queryScaleGm_.SetGlobalBuffer(
                (__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer(
                (__gm__ float *)indexKeyDequantScale);
            sparseAndTailSlotsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)sparseAndTailSlots);
            vectorService_.InitParams(constInfo_);
            vectorService_.InitVecInputTensor(
                weightsGm_, queryScaleGm_, keyScaleGm_,
                sparseAndTailSlotsGm_, indexBlockTableGm_);
            vectorService_.InitVecWorkspaceTensor(scoreWorkspace);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer(
                (__gm__ fp8_e4m3fn_t *)indexKeyCache);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(
                indexBlockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        bool hasSelection = false;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            uint32_t queryStart = 0;
            uint32_t queryEnd = 0;
            if (IsValidSelectionRequest(batch, queryStart, queryEnd)) {
                hasSelection = true;
                break;
            }
        }
        if (!hasSelection) {
            return;
        }

        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            uint32_t queryStart = 0;
            uint32_t queryEnd = 0;
            if (!IsValidSelectionRequest(batch, queryStart, queryEnd)) {
                continue;
            }
            const int32_t mode = rowModesGm_.GetValue(batch);
            for (uint32_t queryRow = queryStart; queryRow < queryEnd;
                 ++queryRow) {
                const uint32_t finalLen = static_cast<uint32_t>(
                    finalSeqLengthsKvGm_.GetValue(batch));
                const uint32_t candidate =
                    mode == ROW_MODE_DENSE
                    ? QueryVisibleLen(queryRow, queryEnd, finalLen)
                    : static_cast<uint32_t>(candidateLensGm_.GetValue(batch));
                if (candidate <= TOPK) {
                    continue;
                }
                const uint32_t loopCount =
                    (candidate + BLOCK_SIZE - 1U) / BLOCK_SIZE;
                for (uint32_t s2 = 0; s2 < loopCount;
                     ++s2, ++globalLoop) {
                    RunInfo run{};
                    run.loop = globalLoop;
                    run.bN2Idx = batch;
                    run.bIdx = batch;
                    run.n2Idx = 0;
                    run.gS1Idx = 0;
                    run.s2Idx = s2;
                    run.actS1Size = 1;
                    run.actS2Size = candidate;
                    run.actS2SizeOrig = candidate;
                    run.actMBaseSize = tiling_->indexHeads;
                    run.actualSingleProcessSInnerSize =
                        candidate - s2 * BLOCK_SIZE < BLOCK_SIZE
                        ? candidate - s2 * BLOCK_SIZE
                        : BLOCK_SIZE;
                    run.actualSingleProcessSInnerSizeAlign = QLICommon::Align(
                        run.actualSingleProcessSInnerSize,
                        ConstInfo::BUFFER_SIZE_BYTE_32B);
                    run.tensorQueryOffset =
                        static_cast<uint64_t>(queryRow) *
                        tiling_->indexHeads * HEAD_DIM;
                    run.tensorKeyOffset =
                        static_cast<uint64_t>(s2) * BLOCK_SIZE * HEAD_DIM;
                    run.tensorKeyScaleOffset =
                        static_cast<uint64_t>(s2) * BLOCK_SIZE;
                    run.tensorWeightsOffset =
                        static_cast<uint64_t>(queryRow) *
                        tiling_->weightStride;
                    run.tensorQueryScaleOffset =
                        static_cast<uint64_t>(queryRow) *
                        tiling_->indexHeads;
                    run.indiceOutOffset =
                        static_cast<uint64_t>(queryRow) *
                        ATTENTION_CAPACITY;
                    run.isFirstS2InnerLoop = s2 == 0;
                    run.isLastS2InnerLoop = s2 + 1U == loopCount;
                    run.isAllLoopEnd = false;
                    run.isValid = true;

                    if ASCEND_IS_AIC {
                        matmulService_.ComputeMm1(run);
                    } else {
                        vectorService_.ProcessVec1(run);
                        if (run.isLastS2InnerLoop) {
                            vectorService_.ProcessTopK(run);
                        }
                    }
                }

                // The score workspace is reused for the next query row.
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                        REQUEST_DONE_EVENT);
                } else if ((subBlockIdx_ & 1U) == 0U) {
                    CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                        REQUEST_DONE_EVENT);
                }
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    __aicore__ inline bool GetQueryRange(
        uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd)
    {
        const int32_t end = actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t start = batch == 0
            ? 0 : actualSeqLengthsQueryGm_.GetValue(batch - 1U);
        if (start < 0 || end <= start ||
            end > static_cast<int32_t>(tiling_->totalQueryRows) ||
            end - start > static_cast<int32_t>(
                MAX_QUERIES_PER_REQUEST)) {
            return false;
        }
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        return true;
    }

    __aicore__ inline bool IsValidSelectionRequest(
        uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd)
    {
        if (!GetQueryRange(batch, queryStart, queryEnd)) {
            return false;
        }
        const int32_t mode = rowModesGm_.GetValue(batch);
        const int32_t finalLen = finalSeqLengthsKvGm_.GetValue(batch);
        if ((mode != ROW_MODE_SPARSE && mode != ROW_MODE_DENSE) ||
            finalLen < static_cast<int32_t>(queryEnd - queryStart) ||
            finalLen > static_cast<int32_t>(tiling_->tokenCapacity)) {
            return false;
        }
        const uint32_t finalLenU32 = static_cast<uint32_t>(finalLen);
        const uint32_t candidate = mode == ROW_MODE_DENSE
            ? QueryVisibleLen(queryEnd - 1U, queryEnd, finalLenU32)
            : static_cast<uint32_t>(candidateLensGm_.GetValue(batch));
        if (mode == ROW_MODE_SPARSE &&
            (candidate % BLOCK_SIZE != 0U ||
             candidate > finalLenU32 - (queryEnd - queryStart - 1U) ||
             finalLenU32 - candidate > 2U * BLOCK_SIZE)) {
            return false;
        }
        return candidate > TOPK &&
            candidate <= tiling_->maxCandidateLen;
    }

    __aicore__ inline uint32_t QueryVisibleLen(
        uint32_t queryRow, uint32_t queryEnd,
        uint32_t finalLen) const
    {
        const uint32_t laterQueries = queryEnd - 1U - queryRow;
        return finalLen > laterQueries ? finalLen - laterQueries : 0;
    }

private:
    TPipe *pipe_;
    const VllmA5LiManageC8TilingData *tiling_;
    uint32_t subBlockIdx_ = 0;
    uint32_t aiCoreIdx_ = 0;
    ConstInfo constInfo_{};
    QLIMatmul<MtpQliType> matmulService_;
    QLIVector<MtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> indexBlockTableGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> finalSeqLengthsKvGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> sparseAndTailSlotsGm_;
};
} // namespace

extern "C" __global__ __aicore__ void vllm_a5_li_manage_c8(
    GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR indexKeyCache,
    GM_ADDR indexKeyDequantScale, GM_ADDR indexBlockTable,
    GM_ADDR candidateLens, GM_ADDR finalSeqLengthsKv, GM_ADDR rowModes,
    GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
    GM_ADDR sparseAndTailSlots, GM_ADDR sparseAndTailSrcIds,
    GM_ADDR perQueryMissCounts, GM_ADDR residentSeqLengths,
    GM_ADDR copySrcIds, GM_ADDR copyDstSlots, GM_ADDR copyCounts,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(VllmA5LiManageC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    const bool requiresColdPath =
        vllm_a5_li_manage_c8_fast::RequiresColdPath(
            actualSeqLengthsQuery, candidateLens, finalSeqLengthsKv,
            rowModes, reqPoolEntries, cacheSlotsPool, &tilingData);
    SynchronizePathDecision();
    if (!requiresColdPath) {
        const uint64_t scoreStride = tilingData.fastScoreWorkspaceStride;
        const uint64_t batchSize = tilingData.batchSize;
        // Keep the workspace helper arithmetic expanded in device code: the
        // constexpr helpers are also consumed by host tiling and are not
        // declared as __aicore__ functions.
        const uint64_t routePairOffset = scoreStride * batchSize;
        const uint64_t routeThresholdOffset =
            routePairOffset +
            batchSize *
                vllm_a5_li_manage_c8_fast_workspace::UNION_CAPACITY *
                2U * sizeof(int32_t);
        const uint64_t routeCountOffset =
            routeThresholdOffset +
            batchSize * vllm_a5_li_manage_c8_fast_workspace::ROUTES *
                vllm_a5_li_manage_c8_fast_workspace::THRESHOLD_STRIDE *
                sizeof(uint16_t);
        GM_ADDR routePairRows = userWorkspace + routePairOffset;
        GM_ADDR routeThresholds = userWorkspace + routeThresholdOffset;
        GM_ADDR routeMissCounts = userWorkspace + routeCountOffset;
        // Stable Stage 1 writes the sparse slot prefixes directly into the
        // caller-owned ABI output. No private full-TopK slot workspace is
        // needed; Stage 2 repairs miss prefixes in place and appends tails.
        GM_ADDR topkSlots = sparseAndTailSlots;

        vllm_a5_li_manage_c8_fast::QuantLiMtpPhase qli(
            &pipe, &tilingData);
        qli.Init(
            indexWeights, query, queryDequantScale,
            actualSeqLengthsQuery, indexKeyCache, indexKeyDequantScale,
            finalSeqLengthsKv,
            reqPoolEntries, cacheSlotsPool, candidateLens, rowModes,
            indexBlockTable, routePairRows, topkSlots,
            sparseAndTailSrcIds,
            routeThresholds, routeMissCounts,
            userWorkspace);
        qli.Process();

        if ASCEND_IS_AIV {
            AscendC::SyncAll();
            if ((GetBlockIdx() & 1U) == 0U) {
                pipe.Reset();
                vllm_a5_li_manage_c8_fast::OrderedMissUnion unionOp;
                unionOp.Init(
                    routePairRows, routeThresholds, routeMissCounts,
                    userWorkspace, actualSeqLengthsQuery,
                    candidateLens, finalSeqLengthsKv,
                    rowModes, reqPoolEntries, cacheSlotsPool, copySrcIds,
                    copyDstSlots, copyCounts, topkSlots,
                    sparseAndTailSlots, sparseAndTailSrcIds,
                    perQueryMissCounts, residentSeqLengths,
                    tilingData.tokenCapacity, tilingData.outputCapacity,
                    tilingData.fastScoreRowStride,
                    tilingData.batchSize, tilingData.poolSize,
                    tilingData.totalQueryRows, tilingData.maxCandidateLen,
                    &pipe);
                unionOp.Process(
                    GetBlockIdx() / 2U, tilingData.usedCoreNum);
            }
        }
        return;
    }

    VllmA5MtpC8QliPhase qli(&pipe, &tilingData);
    qli.Init(
        indexWeights, query, queryDequantScale,
        actualSeqLengthsQuery, indexKeyCache,
        indexKeyDequantScale, indexBlockTable, candidateLens,
        finalSeqLengthsKv,
        rowModes, sparseAndTailSlots, userWorkspace);
    qli.Process();

    pipe.Reset();
    if ASCEND_IS_AIV {
        if ((GetBlockIdx() & 1U) == 0U) {
            vllm_a5_mtp_manager::VllmA5LiManageC8Kernel manager(
                &pipe, &tilingData);
            manager.Init(
                sparseAndTailSlots, actualSeqLengthsQuery,
                candidateLens, finalSeqLengthsKv, rowModes,
                reqPoolEntries, cacheSlotsPool, sparseAndTailSlots,
                sparseAndTailSrcIds, perQueryMissCounts,
                residentSeqLengths, copySrcIds, copyDstSlots,
                copyCounts, GetBlockIdx() / 2U, ATTENTION_CAPACITY);
            manager.Process();
        }
    }
}
