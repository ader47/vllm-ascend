#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace {
constexpr size_t TOPK_INDICES = 0;
constexpr int64_t OUTPUT_CAPACITY = 16384;
}  // namespace

namespace ops {
static ge::graphStatus InferVllmA5LiManageC8Shape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(TOPK_INDICES) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch =
        context->GetInputShape(TOPK_INDICES)->GetDim(0);
    for (size_t index = 0; index < 4; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    *context->GetOutputShape(0) =
        gert::Shape({batch, 1, OUTPUT_CAPACITY});
    *context->GetOutputShape(1) =
        gert::Shape({batch, 1, OUTPUT_CAPACITY});
    *context->GetOutputShape(2) = gert::Shape({batch});
    *context->GetOutputShape(3) = gert::Shape({batch, 2});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferVllmA5LiManageC8DataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(VllmA5LiManageC8)
    .InferShape(InferVllmA5LiManageC8Shape)
    .InferDataType(InferVllmA5LiManageC8DataType);
}  // namespace ops
