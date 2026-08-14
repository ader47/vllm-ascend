/** Tiling registration for the A5 packed-C8 scatter copy. */

#include <cstddef>
#include <cstdint>
#include <limits>

#include "../op_kernel/vllm_a5_kvcache_scatter_copy_c8_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t HBM_KV = 0;
constexpr size_t DRAM_KV = 1;
constexpr size_t HBM_BLOCK_TABLE = 2;
constexpr size_t DRAM_BLOCK_TABLE = 3;
constexpr size_t COPY_SRC_IDS = 4;
constexpr size_t COPY_DST_SLOTS = 5;
constexpr size_t COPY_COUNTS = 6;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t KV_HEADS = 1;
constexpr int64_t PACKED_ROW_BYTES = 656;
constexpr int64_t COPY_CAP = 16384;
constexpr int64_t MAX_SOURCE_TOKENS = 1 << 18;

bool IsPackedCache(const gert::Shape &shape)
{
    return shape.GetDimNum() == 4 &&
        shape.GetDim(1) == BLOCK_SIZE &&
        shape.GetDim(2) == KV_HEADS &&
        shape.GetDim(3) == PACKED_ROW_BYTES;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingVllmA5KvcacheScatterCopyC8(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = HBM_KV; index <= COPY_COUNTS; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context->GetInputDesc(HBM_KV)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DRAM_KV)->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = HBM_BLOCK_TABLE; index <= COPY_COUNTS; ++index) {
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
    const gert::Shape copySrcIds =
        context->GetInputShape(COPY_SRC_IDS)->GetStorageShape();
    const gert::Shape copyDstSlots =
        context->GetInputShape(COPY_DST_SLOTS)->GetStorageShape();
    const gert::Shape copyCounts =
        context->GetInputShape(COPY_COUNTS)->GetStorageShape();
    if (!IsPackedCache(hbmKv) || !IsPackedCache(dramKv) ||
        hbmTable.GetDimNum() != 2 || dramTable.GetDimNum() != 2 ||
        copyCounts.GetDimNum() != 1 || copySrcIds.GetDimNum() != 3 ||
        copySrcIds.GetDim(1) != 1 || copySrcIds.GetDim(2) != COPY_CAP ||
        copyDstSlots.GetDimNum() != 3 ||
        copyDstSlots.GetDim(1) != 1 ||
        copyDstSlots.GetDim(2) != COPY_CAP) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = copyCounts.GetDim(0);
    if (batch <= 0 || hbmKv.GetDim(0) <= 0 || dramKv.GetDim(0) <= 0 ||
        static_cast<uint64_t>(hbmKv.GetDim(0)) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(dramKv.GetDim(0)) >
            std::numeric_limits<uint32_t>::max() ||
        copySrcIds.GetDim(0) != batch ||
        copyDstSlots.GetDim(0) != batch ||
        hbmTable.GetDim(0) != batch ||
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
    tiling->hbmPhysicalBlockCount =
        static_cast<uint32_t>(hbmKv.GetDim(0));
    tiling->dramPhysicalBlockCount =
        static_cast<uint32_t>(dramKv.GetDim(0));
    tiling->packedRowBytes = PACKED_ROW_BYTES;
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
} // namespace optiling
