#include <cstddef>
#include <cstdint>

#include "../op_kernel/vllm_a5_kvcache_scatter_copy_c8_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t HBM_KV = 0;
constexpr size_t DRAM_KV = 1;
constexpr size_t HBM_BLOCK_TABLE = 2;
constexpr size_t DRAM_BLOCK_TABLE = 3;
constexpr size_t SOURCE_TOKEN_IDS = 4;
constexpr size_t DESTINATION_SLOTS = 5;
constexpr size_t COPY_COUNTS = 6;
constexpr size_t CACHE_TOKENS = 7;
constexpr size_t CANDIDATE_LENS = 8;
constexpr size_t ACTUAL_SEQ_LENGTHS_KV = 9;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t KV_HEADS = 1;
constexpr int64_t PACKED_ROW_BYTES = 656;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t COPY_CAP = 16384;
constexpr int64_t TAIL_CAP = BLOCK_SIZE;
constexpr int64_t ATTENTION_CAP = SPARSE_COUNT + TAIL_CAP;
constexpr int64_t MAX_SOURCE_TOKENS = 1 << 18;

bool IsPackedCache(const gert::Shape &shape)
{
    return shape.GetDimNum() == 4 &&
        shape.GetDim(0) > 0 &&
        shape.GetDim(1) == BLOCK_SIZE &&
        shape.GetDim(2) == KV_HEADS &&
        shape.GetDim(3) == PACKED_ROW_BYTES;
}

bool GetMetadataShape(
    const gert::Shape &shape,
    int64_t &batch,
    int64_t &capacity)
{
    if (shape.GetDimNum() == 2) {
        batch = shape.GetDim(0);
        capacity = shape.GetDim(1);
        return true;
    }
    if (shape.GetDimNum() == 3 && shape.GetDim(1) == 1) {
        batch = shape.GetDim(0);
        capacity = shape.GetDim(2);
        return true;
    }
    return false;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingVllmA5KvcacheScatterCopyC8(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = HBM_KV; index <= ACTUAL_SEQ_LENGTHS_KV; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context->GetInputDesc(HBM_KV)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DRAM_KV)->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = HBM_BLOCK_TABLE;
         index <= ACTUAL_SEQ_LENGTHS_KV; ++index) {
        if (context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape hbmKv =
        context->GetInputShape(HBM_KV)->GetStorageShape();
    const gert::Shape dramKv =
        context->GetInputShape(DRAM_KV)->GetStorageShape();
    const gert::Shape hbmTable =
        context->GetInputShape(HBM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape dramTable =
        context->GetInputShape(DRAM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape sourceIds =
        context->GetInputShape(SOURCE_TOKEN_IDS)->GetStorageShape();
    const gert::Shape destinationSlots =
        context->GetInputShape(DESTINATION_SLOTS)->GetStorageShape();
    const gert::Shape copyCounts =
        context->GetInputShape(COPY_COUNTS)->GetStorageShape();
    const gert::Shape cacheTokens =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape candidateLens =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    const gert::Shape actualKv =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_KV)->GetStorageShape();

    int64_t sourceBatch = 0;
    int64_t sourceCapacity = 0;
    int64_t destinationBatch = 0;
    int64_t destinationCapacity = 0;
    if (!IsPackedCache(hbmKv) || !IsPackedCache(dramKv) ||
        hbmTable.GetDimNum() != 2 || dramTable.GetDimNum() != 2 ||
        copyCounts.GetDimNum() != 1 || cacheTokens.GetDimNum() != 1 ||
        candidateLens.GetDimNum() != 1 || actualKv.GetDimNum() != 1 ||
        !GetMetadataShape(sourceIds, sourceBatch, sourceCapacity) ||
        !GetMetadataShape(
            destinationSlots, destinationBatch, destinationCapacity)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = copyCounts.GetDim(0);
    if (batch <= 0 || sourceBatch != batch || destinationBatch != batch ||
        sourceCapacity != COPY_CAP || destinationCapacity != COPY_CAP ||
        cacheTokens.GetDim(0) != batch || candidateLens.GetDim(0) != batch ||
        actualKv.GetDim(0) != batch || hbmTable.GetDim(0) != batch ||
        dramTable.GetDim(0) != batch || hbmTable.GetDim(1) <= 0 ||
        dramTable.GetDim(1) <= 0 ||
        dramTable.GetDim(1) * BLOCK_SIZE > MAX_SOURCE_TOKENS) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalPairSlots =
        static_cast<uint64_t>(batch) * COPY_CAP;
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        totalPairSlots < aivCount ? totalPairSlots : aivCount);
    auto *tiling = context->GetTilingData<
        VllmA5KvcacheScatterCopyC8TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->copyCap = COPY_CAP;
    tiling->hbmMaxBlockNum = static_cast<uint32_t>(hbmTable.GetDim(1));
    tiling->dramMaxBlockNum = static_cast<uint32_t>(dramTable.GetDim(1));
    tiling->hbmPhysicalBlockCount = static_cast<uint32_t>(hbmKv.GetDim(0));
    tiling->dramPhysicalBlockCount = static_cast<uint32_t>(dramKv.GetDim(0));
    tiling->packedRowBytes = PACKED_ROW_BYTES;
    tiling->attentionCapacity = ATTENTION_CAP;
    tiling->totalPairSlots = totalPairSlots;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

struct VllmA5KvcacheScatterCopyC8CompileInfo {};

static ge::graphStatus TilingParseVllmA5KvcacheScatterCopyC8(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmA5KvcacheScatterCopyC8)
    .Tiling(TilingVllmA5KvcacheScatterCopyC8)
    .TilingParse<VllmA5KvcacheScatterCopyC8CompileInfo>(
        TilingParseVllmA5KvcacheScatterCopyC8);
}  // namespace optiling
