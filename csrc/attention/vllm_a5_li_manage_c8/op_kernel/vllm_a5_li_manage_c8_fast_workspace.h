#ifndef VLLM_A5_LI_MANAGE_C8_FAST_WORKSPACE_H
#define VLLM_A5_LI_MANAGE_C8_FAST_WORKSPACE_H

#include <cstdint>

namespace vllm_a5_li_manage_c8_fast_workspace {

constexpr uint64_t ROUTES = 4U;
constexpr uint64_t TOPK = 2048U;
constexpr uint64_t UNION_CAPACITY = ROUTES * TOPK;
constexpr uint64_t THRESHOLD_STRIDE = 16U;
constexpr uint64_t ATTENTION_CAPACITY = TOPK + 128U;
constexpr uint64_t ALIGN_BYTES = 32U;

constexpr uint64_t AlignBytes(uint64_t value)
{
    return (value + ALIGN_BYTES - 1U) / ALIGN_BYTES * ALIGN_BYTES;
}

constexpr uint64_t ScoreBytes(uint64_t scoreStrideBytes, uint64_t batch)
{
    return scoreStrideBytes * batch;
}

constexpr uint64_t RoutePairOffset(
    uint64_t scoreStrideBytes, uint64_t batch)
{
    return ScoreBytes(scoreStrideBytes, batch);
}

constexpr uint64_t RoutePairBytes(uint64_t batch)
{
    return batch * UNION_CAPACITY * 2U * sizeof(int32_t);
}

constexpr uint64_t RouteThresholdOffset(
    uint64_t scoreStrideBytes, uint64_t batch)
{
    return RoutePairOffset(scoreStrideBytes, batch) +
        RoutePairBytes(batch);
}

constexpr uint64_t RouteThresholdBytes(uint64_t batch)
{
    return batch * ROUTES * THRESHOLD_STRIDE * sizeof(uint16_t);
}

constexpr uint64_t RouteCountOffset(
    uint64_t scoreStrideBytes, uint64_t batch)
{
    return RouteThresholdOffset(scoreStrideBytes, batch) +
        RouteThresholdBytes(batch);
}

constexpr uint64_t RouteCountBytes(uint64_t batch)
{
    return batch * ROUTES * sizeof(int32_t);
}

constexpr uint64_t TotalBytes(
    uint64_t scoreStrideBytes, uint64_t batch)
{
    return AlignBytes(
        RouteCountOffset(scoreStrideBytes, batch) +
        RouteCountBytes(batch));
}

} // namespace vllm_a5_li_manage_c8_fast_workspace

#endif
