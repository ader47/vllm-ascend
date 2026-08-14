/**
 * A5 non-MTP C8 LightningIndexer + request-pool management.
 *
 * This is deliberately one MIX_AIC_1_2 kernel.  The first phase reuses the
 * native A5 C8 quant-LI Cube/Vector/TopK services.  After TopK is materialized
 * into the caller-owned attention row, the even AIV resets its TPipe and runs
 * the correctness-first request-pool manager in place.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "vllm_a5_li_manage_nomtp_c8_tiling.h"
#include "arch35/quant_lightning_indexer_common.h"
#include "arch35/quant_lightning_indexer_service_cube.h"
#include "arch35/quant_lightning_indexer_service_vector.h"
#include "vllm_a5_li_manage_nomtp_c8_manager.h"

namespace {
using namespace AscendC;
using namespace QLICommon;
using namespace QLIKernel;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t TOPK = 2048;
constexpr uint32_t ATTENTION_CAPACITY = TOPK + BLOCK_SIZE;
constexpr int32_t ROW_MODE_DENSE = 1;
constexpr int32_t ROW_MODE_SPARSE = 2;
constexpr uint32_t REQUEST_DONE_EVENT = 6;

using NomtpQliType = QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, true,
    LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>;

class VllmA5NomtpC8QliPhase {
public:
    __aicore__ inline VllmA5NomtpC8QliPhase(
        TPipe *pipe, const VllmA5LiManageNomtpC8TilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
        GM_ADDR indexKeyCache, GM_ADDR indexKeyDequantScale,
        GM_ADDR indexBlockTable, GM_ADDR candidateLens, GM_ADDR rowModes,
        GM_ADDR sparseAndTailSlots, GM_ADDR userWorkspace)
    {
        if ASCEND_IS_AIV {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_;
        }

        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
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
            const int32_t mode = rowModesGm_.GetValue(batch);
            const int32_t candidate = candidateLensGm_.GetValue(batch);
            const bool sparseSelection =
                mode == ROW_MODE_SPARSE &&
                candidate >= static_cast<int32_t>(TOPK) &&
                candidate % static_cast<int32_t>(BLOCK_SIZE) == 0;
            const bool denseSelection =
                mode == ROW_MODE_DENSE &&
                candidate > static_cast<int32_t>(TOPK);
            if ((sparseSelection || denseSelection) &&
                candidate <= static_cast<int32_t>(
                    tiling_->maxCandidateLen)) {
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
            const int32_t mode = rowModesGm_.GetValue(batch);
            const int32_t candidateSigned = candidateLensGm_.GetValue(batch);
            const bool sparseSelection =
                mode == ROW_MODE_SPARSE &&
                candidateSigned >= static_cast<int32_t>(TOPK) &&
                candidateSigned % static_cast<int32_t>(BLOCK_SIZE) == 0;
            const bool denseSelection =
                mode == ROW_MODE_DENSE &&
                candidateSigned > static_cast<int32_t>(TOPK);
            if ((!sparseSelection && !denseSelection) ||
                candidateSigned > static_cast<int32_t>(
                    tiling_->maxCandidateLen)) {
                continue;
            }
            const uint32_t candidate =
                static_cast<uint32_t>(candidateSigned);
            const uint32_t loopCount =
                (candidate + BLOCK_SIZE - 1U) / BLOCK_SIZE;
            for (uint32_t s2 = 0; s2 < loopCount; ++s2, ++globalLoop) {
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
                    static_cast<uint64_t>(batch) * tiling_->indexHeads *
                    HEAD_DIM;
                run.tensorKeyOffset =
                    static_cast<uint64_t>(s2) * BLOCK_SIZE * HEAD_DIM;
                run.tensorKeyScaleOffset =
                    static_cast<uint64_t>(s2) * BLOCK_SIZE;
                run.tensorWeightsOffset =
                    static_cast<uint64_t>(batch) * tiling_->weightStride;
                // weights may be a strided suffix view of wk_weights_proj,
                // while query_dequant_scale is a compact [B, N_idx] tensor.
                // Their row offsets must therefore be tracked independently.
                run.tensorQueryScaleOffset =
                    static_cast<uint64_t>(batch) * tiling_->indexHeads;
                run.indiceOutOffset =
                    static_cast<uint64_t>(batch) * ATTENTION_CAPACITY;
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

            // Do not let the Cube start overwriting this group's score
            // workspace for the next request before the even AIV has consumed
            // it in TopK.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                    REQUEST_DONE_EVENT);
            } else if ((subBlockIdx_ & 1U) == 0U) {
                CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                    REQUEST_DONE_EVENT);
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
        } else {
            matmulService_.FreeEventID();
            // Consume any initial ping-pong credits that were not used by the
            // final request so no cross-core state leaks past this phase.
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    TPipe *pipe_;
    const VllmA5LiManageNomtpC8TilingData *tiling_;
    uint32_t subBlockIdx_ = 0;
    uint32_t aiCoreIdx_ = 0;
    ConstInfo constInfo_{};
    QLIMatmul<NomtpQliType> matmulService_;
    QLIVector<NomtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> indexBlockTableGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> sparseAndTailSlotsGm_;
};
} // namespace

extern "C" __global__ __aicore__ void vllm_a5_li_manage_nomtp_c8(
    GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR indexKeyCache,
    GM_ADDR indexKeyDequantScale, GM_ADDR indexBlockTable,
    GM_ADDR candidateLens, GM_ADDR finalSeqLengthsKv, GM_ADDR rowModes,
    GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
    GM_ADDR sparseAndTailSlots, GM_ADDR residentSeqLengths,
    GM_ADDR copySrcIds, GM_ADDR copyDstSlots, GM_ADDR copyCounts,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(VllmA5LiManageNomtpC8TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    VllmA5NomtpC8QliPhase qli(&pipe, &tilingData);
    qli.Init(
        indexWeights, query, queryDequantScale, indexKeyCache,
        indexKeyDequantScale, indexBlockTable, candidateLens, rowModes,
        sparseAndTailSlots, userWorkspace);
    qli.Process();

    pipe.Reset();
    if ASCEND_IS_AIV {
        if ((GetBlockIdx() & 1U) == 0U) {
            vllm_a5_nomtp_manager::VllmA5LiManageNomtpC8Manager manager(
                &pipe, &tilingData);
            // TopK was written into the first 2048 entries of each
            // sparse_and_tail_slots row.  The manager loads it before
            // replacing that same row with cache slots + causal tail.
            manager.Init(
                sparseAndTailSlots, actualSeqLengthsQuery,
                candidateLens, finalSeqLengthsKv, rowModes,
                reqPoolEntries, cacheSlotsPool, sparseAndTailSlots,
                residentSeqLengths, copySrcIds, copyDstSlots,
                copyCounts);
            manager.Process();
        }
    }
}
