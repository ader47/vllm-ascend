#include <cstddef>

#include "register/op_impl_registry.h"

namespace {
constexpr size_t DST_CACHE = 1;
}  // namespace

namespace ops {
static ge::graphStatus InferKvCacheFullBlockDumpC8Shape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(DST_CACHE) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(DST_CACHE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferKvCacheFullBlockDumpC8DataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheFullBlockDumpC8)
    .InferShape(InferKvCacheFullBlockDumpC8Shape)
    .InferDataType(InferKvCacheFullBlockDumpC8DataType);
}  // namespace ops
