/** Shape and dtype inference for the A5 packed-C8 scatter copy. */

#include <cstddef>

#include "register/op_impl_registry.h"

namespace {
constexpr size_t HBM_KV = 0;
} // namespace

namespace ops {
static ge::graphStatus InferVllmA5KvcacheScatterCopyC8Shape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(HBM_KV) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(HBM_KV);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferVllmA5KvcacheScatterCopyC8DataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(VllmA5KvcacheScatterCopyC8)
    .InferShape(InferVllmA5KvcacheScatterCopyC8Shape)
    .InferDataType(InferVllmA5KvcacheScatterCopyC8DataType);
} // namespace ops
