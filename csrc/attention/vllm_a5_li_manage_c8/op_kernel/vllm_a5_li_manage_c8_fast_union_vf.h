#ifndef VLLM_A5_LI_MANAGE_C8_FAST_UNION_VF_H
#define VLLM_A5_LI_MANAGE_C8_FAST_UNION_VF_H

#include "kernel_operator.h"

namespace VllmA5LiManageC8FastUnionVF {
using namespace AscendC;

constexpr uint32_t LANES = 64U;
constexpr uint32_t TAIL_CAPACITY = 256U;

// MrgSort emits sorted (key, source) pairs. Compare each source with its
// global predecessor, including at vector boundaries, then compact once.
// Invalid lanes gather pair zero, never beyond the last valid input pair.
// No packed GatherMask predicate/repeat geometry or scalar O(total) walk.
__simd_vf__ void DeduplicateSortedPairs(
    __ubuf__ uint32_t *dst, __ubuf__ uint32_t *pairs, uint32_t total)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> zero;
    MicroAPI::RegTensor<uint32_t> one;
    MicroAPI::RegTensor<uint32_t> limit;
    MicroAPI::RegTensor<uint32_t> sentinel;
    MicroAPI::Duplicate(zero, 0U);
    MicroAPI::Duplicate(one, 1U);
    MicroAPI::Duplicate(limit, total);
    MicroAPI::Duplicate(sentinel, 0xffffffffU); // outside the 18-bit source domain
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t base = 0U; base < total; base += LANES) {
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(base));
        auto &position = (MicroAPI::RegTensor<uint32_t>&)positions;
        MicroAPI::MaskReg valid;
        MicroAPI::MaskReg first;
        MicroAPI::MaskReg hasPrevious;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(valid, position, limit, all);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(first, position, zero, all);
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(hasPrevious, position, zero, all);
        MicroAPI::And(hasPrevious, hasPrevious, valid, all);

        MicroAPI::RegTensor<uint32_t> currentOffset;
        MicroAPI::RegTensor<uint32_t> previousOffset;
        MicroAPI::Select(currentOffset, position, zero, valid);
        MicroAPI::Sub(previousOffset, position, one, all);
        MicroAPI::Select(previousOffset, previousOffset, zero, hasPrevious);
        MicroAPI::ShiftLefts(currentOffset, currentOffset, static_cast<int16_t>(1), all);
        MicroAPI::ShiftLefts(previousOffset, previousOffset, static_cast<int16_t>(1), all);
        MicroAPI::Add(currentOffset, currentOffset, one, all);
        MicroAPI::Add(previousOffset, previousOffset, one, all);

        MicroAPI::RegTensor<uint32_t> current;
        MicroAPI::RegTensor<uint32_t> previous;
        MicroAPI::Gather(current, pairs, currentOffset, all);
        MicroAPI::Gather(previous, pairs, previousOffset, all);
        MicroAPI::Select(previous, sentinel, previous, first);
        MicroAPI::MaskReg keep;
        MicroAPI::Compare<uint32_t, CMPMODE::NE>(keep, current, previous, all);
        MicroAPI::And(keep, keep, valid, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compact, current, keep);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
}

// Each route has its own aligned UB row: MTE3 can publish all routes without
// waiting between them or reusing a row that is still being read by DMA.
__simd_vf__ void BuildCausalTailRows(
    __ubuf__ uint32_t *slots, __ubuf__ uint32_t *sources,
    uint32_t candidate, uint32_t budget, uint32_t finalLen,
    uint32_t routeCount)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> invalid;
    MicroAPI::RegTensor<uint32_t> tailMask;
    MicroAPI::RegTensor<uint32_t> cacheBudget;
    MicroAPI::Duplicate(invalid, 0xffffffffU);
    MicroAPI::Duplicate(tailMask, TAIL_CAPACITY - 1U);
    MicroAPI::Duplicate(cacheBudget, budget);
    for (uint32_t route = 0U; route < routeCount; ++route) {
        MicroAPI::RegTensor<uint32_t> visible;
        MicroAPI::Duplicate(visible, finalLen - (routeCount - 1U - route));
        for (uint32_t offset = 0U; offset < TAIL_CAPACITY; offset += LANES) {
            MicroAPI::RegTensor<int32_t> tokens;
            MicroAPI::Arange(tokens, static_cast<int32_t>(candidate + offset));
            auto &token = (MicroAPI::RegTensor<uint32_t>&)tokens;
            MicroAPI::MaskReg valid;
            MicroAPI::Compare<uint32_t, CMPMODE::LT>(valid, token, visible, all);
            MicroAPI::RegTensor<uint32_t> slot;
            MicroAPI::RegTensor<uint32_t> source;
            MicroAPI::And(slot, token, tailMask, all);
            MicroAPI::Add(slot, slot, cacheBudget, all);
            MicroAPI::Select(slot, slot, invalid, valid);
            MicroAPI::Select(source, token, invalid, valid);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
                slots + route * TAIL_CAPACITY + offset, slot, all);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
                sources + route * TAIL_CAPACITY + offset, source, all);
        }
    }
}

} // namespace VllmA5LiManageC8FastUnionVF

#endif
