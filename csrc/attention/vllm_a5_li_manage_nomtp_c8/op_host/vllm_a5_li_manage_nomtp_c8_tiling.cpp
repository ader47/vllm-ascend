/** Tiling registration for the A5 non-MTP C8 LI + manage op. */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "../op_kernel/vllm_a5_li_manage_nomtp_c8_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    INDEX_WEIGHTS = 0,
    QUERY,
    QUERY_DEQUANT_SCALE,
    ACTUAL_SEQ_LENGTHS_QUERY,
    INDEX_KEY_CACHE,
    INDEX_KEY_DEQUANT_SCALE,
    INDEX_BLOCK_TABLE,
    CANDIDATE_LENS,
    FINAL_SEQ_LENGTHS_KV,
    ROW_MODES,
    REQ_POOL_ENTRIES,
    CACHE_SLOTS_POOL,
};

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t OUTPUT_CAPACITY = 16384;
constexpr int64_t MAX_TOKEN_CAPACITY = 1 << 18;
constexpr size_t ATTR_KEY_STRIDE = 0;
constexpr size_t ATTR_SCALE_STRIDE = 1;
constexpr size_t ATTR_WEIGHT_STRIDE = 2;

bool IsShape(
    const gert::Shape &shape, std::initializer_list<int64_t> dimensions)
{
    if (shape.GetDimNum() != dimensions.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t expected : dimensions) {
        if (expected >= 0 && shape.GetDim(index) != expected) {
            return false;
        }
        ++index;
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingVllmA5LiManageNomtpC8(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index <= CACHE_SLOTS_POOL; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const auto dtype = [context](size_t index) {
        return context->GetInputDesc(index)->GetDataType();
    };
    if (dtype(INDEX_WEIGHTS) != ge::DT_BF16 ||
        dtype(QUERY) != ge::DT_FLOAT8_E4M3FN ||
        dtype(QUERY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(INDEX_KEY_CACHE) != ge::DT_FLOAT8_E4M3FN ||
        dtype(INDEX_KEY_DEQUANT_SCALE) != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {
             ACTUAL_SEQ_LENGTHS_QUERY, INDEX_BLOCK_TABLE, CANDIDATE_LENS,
             FINAL_SEQ_LENGTHS_KV, ROW_MODES, REQ_POOL_ENTRIES,
             CACHE_SLOTS_POOL}) {
        if (dtype(index) != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape weights =
        context->GetInputShape(INDEX_WEIGHTS)->GetStorageShape();
    const gert::Shape query =
        context->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape queryScale =
        context->GetInputShape(QUERY_DEQUANT_SCALE)->GetStorageShape();
    const gert::Shape queryLengths =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const gert::Shape key =
        context->GetInputShape(INDEX_KEY_CACHE)->GetStorageShape();
    const gert::Shape keyScale =
        context->GetInputShape(INDEX_KEY_DEQUANT_SCALE)->GetStorageShape();
    const gert::Shape blockTable =
        context->GetInputShape(INDEX_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape candidate =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    const gert::Shape finalLengths =
        context->GetInputShape(FINAL_SEQ_LENGTHS_KV)->GetStorageShape();
    const gert::Shape modes =
        context->GetInputShape(ROW_MODES)->GetStorageShape();
    const gert::Shape entries =
        context->GetInputShape(REQ_POOL_ENTRIES)->GetStorageShape();
    const gert::Shape pool =
        context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();

    if (!IsShape(weights, {-1, -1}) ||
        !IsShape(query, {-1, -1, HEAD_DIM}) ||
        !IsShape(queryScale, {-1, -1}) ||
        !IsShape(queryLengths, {-1}) ||
        !IsShape(key, {-1, BLOCK_SIZE, 1, HEAD_DIM}) ||
        !IsShape(keyScale, {-1, BLOCK_SIZE, 1}) ||
        !IsShape(blockTable, {-1, -1}) || !IsShape(candidate, {-1}) ||
        !IsShape(finalLengths, {-1}) || !IsShape(modes, {-1}) ||
        !IsShape(entries, {-1}) || !IsShape(pool, {-1, -1})) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = query.GetDim(0);
    const int64_t heads = query.GetDim(1);
    if (batch <= 0 || (heads != 32 && heads != 64) ||
        weights.GetDim(0) != batch || weights.GetDim(1) != heads ||
        queryScale.GetDim(0) != batch || queryScale.GetDim(1) != heads ||
        queryLengths.GetDim(0) != batch || blockTable.GetDim(0) != batch ||
        candidate.GetDim(0) != batch || finalLengths.GetDim(0) != batch ||
        modes.GetDim(0) != batch || entries.GetDim(0) != batch ||
        key.GetDim(0) <= 0 || keyScale.GetDim(0) != key.GetDim(0) ||
        pool.GetDim(0) <= 0 || pool.GetDim(1) < 2 ||
        pool.GetDim(1) > MAX_TOKEN_CAPACITY + 1 ||
        blockTable.GetDim(1) <= 0 ||
        blockTable.GetDim(1) * BLOCK_SIZE > MAX_TOKEN_CAPACITY) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *keyStride =
        attrs->GetAttrPointer<int64_t>(ATTR_KEY_STRIDE);
    const int64_t *scaleStride =
        attrs->GetAttrPointer<int64_t>(ATTR_SCALE_STRIDE);
    const int64_t *weightStride =
        attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_STRIDE);
    if (keyStride == nullptr || scaleStride == nullptr ||
        weightStride == nullptr || *weightStride < heads ||
        *keyStride < BLOCK_SIZE * HEAD_DIM || *scaleStride < BLOCK_SIZE ||
        static_cast<uint64_t>(*weightStride) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*keyStride) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*scaleStride) >
            std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCount = platform.GetCoreNumAic();
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aicCount == 0 || aivCount < 2) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = std::min<uint32_t>(
        static_cast<uint32_t>(batch), aicCount);
    const uint32_t maxCandidateLen = static_cast<uint32_t>(
        blockTable.GetDim(1) * BLOCK_SIZE);
    const uint32_t s1BaseSize =
        (256U + static_cast<uint32_t>(heads) - 1U) /
        static_cast<uint32_t>(heads);
    const uint64_t scoreStride64 =
        static_cast<uint64_t>(s1BaseSize) * maxCandidateLen *
        sizeof(uint16_t);
    if (scoreStride64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling =
        context->GetTilingData<VllmA5LiManageNomtpC8TilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->totalQueryRows = static_cast<uint32_t>(batch);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->tokenCapacity = static_cast<uint32_t>(pool.GetDim(1) - 1);
    tiling->outputCapacity = OUTPUT_CAPACITY;
    tiling->indexHeads = static_cast<uint32_t>(heads);
    tiling->maxBlockNumPerBatch =
        static_cast<uint32_t>(blockTable.GetDim(1));
    tiling->maxCandidateLen = maxCandidateLen;
    tiling->weightStride = static_cast<uint32_t>(*weightStride);
    tiling->keyStride = static_cast<uint32_t>(*keyStride);
    tiling->scaleStride = static_cast<uint32_t>(*scaleStride);
    tiling->scoreWorkspaceStride = static_cast<uint32_t>(scoreStride64);

    context->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() +
        scoreStride64 * static_cast<uint64_t>(usedCoreNum);
    context->SetBlockDim(platform.CalcTschBlockDim(
        usedCoreNum * 2U, usedCoreNum, usedCoreNum * 2U));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

struct VllmA5LiManageNomtpC8CompileInfo {};

static ge::graphStatus TilingParseVllmA5LiManageNomtpC8(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmA5LiManageNomtpC8)
    .Tiling(TilingVllmA5LiManageNomtpC8)
    .TilingParse<VllmA5LiManageNomtpC8CompileInfo>(
        TilingParseVllmA5LiManageNomtpC8);
} // namespace optiling
