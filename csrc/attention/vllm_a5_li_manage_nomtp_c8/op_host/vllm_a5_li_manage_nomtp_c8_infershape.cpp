/** Shape and dtype inference for the A5 non-MTP C8 LI + manage op. */

#include <cstddef>
#include <cstdint>

#include "register/op_impl_registry.h"

namespace {
constexpr size_t QUERY = 1;
constexpr int64_t TOPK = 2048;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t ATTENTION_CAPACITY = TOPK + BLOCK_SIZE;
constexpr int64_t OUTPUT_CAPACITY = 16384;
} // namespace

namespace ops {
static ge::graphStatus InferVllmA5LiManageNomtpC8Shape(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(QUERY) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = context->GetInputShape(QUERY)->GetDim(0);
    for (size_t index = 0; index < 5; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    *context->GetOutputShape(0) =
        gert::Shape({batch, 1, ATTENTION_CAPACITY});
    *context->GetOutputShape(1) = gert::Shape({batch});
    *context->GetOutputShape(2) =
        gert::Shape({batch, 1, OUTPUT_CAPACITY});
    *context->GetOutputShape(3) =
        gert::Shape({batch, 1, OUTPUT_CAPACITY});
    *context->GetOutputShape(4) = gert::Shape({batch});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferVllmA5LiManageNomtpC8DataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 5; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(VllmA5LiManageNomtpC8)
    .InferShape(InferVllmA5LiManageNomtpC8Shape)
    .InferDataType(InferVllmA5LiManageNomtpC8DataType);
} // namespace ops
