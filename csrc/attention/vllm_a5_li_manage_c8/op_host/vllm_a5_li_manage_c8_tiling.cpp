#include <cstddef>
#include <cstdint>

#include "../op_kernel/vllm_a5_li_manage_c8_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t TOPK_INDICES = 0;
constexpr size_t REQ_POOL_ENTRIES = 1;
constexpr size_t CACHE_SLOTS_POOL = 2;
constexpr size_t ROW_MODES = 3;
constexpr size_t ACTUAL_SEQ_LENGTHS_KEY = 4;

constexpr int64_t TOPK = 2048;
constexpr int64_t OUTPUT_CAPACITY = 16384;
constexpr int64_t MAX_TOKEN_CAPACITY = 1 << 18;
}  // namespace

namespace optiling {
static ge::graphStatus TilingVllmA5LiManageC8(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = TOPK_INDICES;
         index <= ACTUAL_SEQ_LENGTHS_KEY; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape topk =
        context->GetInputShape(TOPK_INDICES)->GetStorageShape();
    const gert::Shape req =
        context->GetInputShape(REQ_POOL_ENTRIES)->GetStorageShape();
    const gert::Shape pool =
        context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();
    const gert::Shape modes =
        context->GetInputShape(ROW_MODES)->GetStorageShape();
    const gert::Shape lengths =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_KEY)->GetStorageShape();
    if (topk.GetDimNum() != 3 || topk.GetDim(0) <= 0 ||
        topk.GetDim(1) != 1 || topk.GetDim(2) != TOPK ||
        req.GetDimNum() != 1 || modes.GetDimNum() != 1 ||
        lengths.GetDimNum() != 1 || pool.GetDimNum() != 2 ||
        pool.GetDim(0) <= 0 || pool.GetDim(1) < 2 ||
        pool.GetDim(1) > MAX_TOKEN_CAPACITY + 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = topk.GetDim(0);
    if (req.GetDim(0) != batch || modes.GetDim(0) != batch ||
        lengths.GetDim(0) != batch) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        batch < static_cast<int64_t>(aivCount) ? batch : aivCount);
    auto *tiling = context->GetTilingData<VllmA5LiManageC8TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->tokenCapacity = static_cast<uint32_t>(pool.GetDim(1) - 1);
    tiling->outputCapacity = OUTPUT_CAPACITY;
    context->SetBlockDim(usedCoreNum);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

struct VllmA5LiManageC8CompileInfo {};

static ge::graphStatus TilingParseVllmA5LiManageC8(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmA5LiManageC8)
    .Tiling(TilingVllmA5LiManageC8)
    .TilingParse<VllmA5LiManageC8CompileInfo>(
        TilingParseVllmA5LiManageC8);
}  // namespace optiling
