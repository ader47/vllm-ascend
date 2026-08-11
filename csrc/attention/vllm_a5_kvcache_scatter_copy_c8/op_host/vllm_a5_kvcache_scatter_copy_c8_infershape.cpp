#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace {
constexpr size_t HBM_KV = 0;
constexpr size_t COPY_COUNTS = 6;
constexpr int64_t ATTENTION_CAPACITY = 2176;
}  // namespace

namespace ops {
static ge::graphStatus InferKvcacheScatterCopyC8Shape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(HBM_KV) == nullptr ||
        context->GetInputShape(COPY_COUNTS) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = context->GetInputShape(COPY_COUNTS)->GetDim(0);
    *context->GetOutputShape(0) = *context->GetInputShape(HBM_KV);
    *context->GetOutputShape(1) =
        gert::Shape({batch, 1, ATTENTION_CAPACITY});
    *context->GetOutputShape(2) = gert::Shape({batch});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferKvcacheScatterCopyC8DataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(VllmA5KvcacheScatterCopyC8)
    .InferShape(InferKvcacheScatterCopyC8Shape)
    .InferDataType(InferKvcacheScatterCopyC8DataType);
}  // namespace ops
