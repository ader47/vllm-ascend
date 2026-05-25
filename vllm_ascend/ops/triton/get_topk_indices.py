import torch
import triton
import triton.language as tl


@triton.jit
def clear_cache_miss_hash_kernel(
        old_hash_ptr,
        new_hash_ptr,
        num_reqs,
        HASH_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    if pid >= num_reqs:
        return

    cols = bid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < HASH_CAPACITY
    row_off = pid * HASH_CAPACITY
    empty = tl.full((BLOCK,), -1, tl.int32)
    tl.store(old_hash_ptr + row_off + cols, empty, mask=mask)
    tl.store(new_hash_ptr + row_off + cols, empty, mask=mask)


@triton.jit
def mark_cache_tokens_hash_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_hash_ptr,
        new_hash_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        HASH_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
        PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    hash_off = pid * HASH_CAPACITY
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = (old_with_offset - req_offset).to(tl.int32)
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_hash = (old_token * 1103515245 + 12345) & (HASH_CAPACITY - 1)
    old_inserted = tl.full((BLOCK,), False, tl.int1)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_hash = (new_token * 1103515245 + 12345) & (HASH_CAPACITY - 1)
    new_inserted = tl.full((BLOCK,), False, tl.int1)

    for probe_idx in range(PROBE):
        old_slot = (old_hash + probe_idx) & (HASH_CAPACITY - 1)
        old_active = old_valid & ~old_inserted
        old_cmp = tl.where(old_active, -1, -2)
        old_value = tl.where(old_active, old_token, -1)
        old_prev = tl.atomic_cas(
            old_hash_ptr + hash_off + old_slot,
            old_cmp,
            old_value,
            sem="relaxed",
        )
        old_inserted = old_inserted | (old_active & ((old_prev == -1) | (old_prev == old_token)))

        new_slot = (new_hash + probe_idx) & (HASH_CAPACITY - 1)
        new_active = new_valid & ~new_inserted
        new_cmp = tl.where(new_active, -1, -2)
        new_value = tl.where(new_active, new_token, -1)
        new_prev = tl.atomic_cas(
            new_hash_ptr + hash_off + new_slot,
            new_cmp,
            new_value,
            sem="relaxed",
        )
        new_inserted = new_inserted | (new_active & ((new_prev == -1) | (new_prev == new_token)))


@triton.jit
def compact_cache_miss_slots_hash_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_hash_ptr,
        new_hash_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        HASH_CAPACITY: tl.constexpr,
        BLOCK: tl.constexpr,
        PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    hash_off = pid * HASH_CAPACITY
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_hash = (new_token * 1103515245 + 12345) & (HASH_CAPACITY - 1)
    new_found = tl.full((BLOCK,), False, tl.int1)
    new_blocked = tl.full((BLOCK,), False, tl.int1)

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = (old_with_offset - req_offset).to(tl.int32)
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_hash = (old_token * 1103515245 + 12345) & (HASH_CAPACITY - 1)
    old_found = tl.full((BLOCK,), False, tl.int1)
    old_blocked = tl.full((BLOCK,), False, tl.int1)

    for probe_idx in range(PROBE):
        new_slot = (new_hash + probe_idx) & (HASH_CAPACITY - 1)
        old_key = tl.load(
            old_hash_ptr + hash_off + new_slot,
            mask=new_valid & ~new_found & ~new_blocked,
            other=-1,
        )
        new_found = new_found | (new_valid & (old_key == new_token))
        new_blocked = new_blocked | (new_valid & (old_key == -1))

        old_slot = (old_hash + probe_idx) & (HASH_CAPACITY - 1)
        new_key = tl.load(
            new_hash_ptr + hash_off + old_slot,
            mask=old_valid & ~old_found & ~old_blocked,
            other=-1,
        )
        old_found = old_found | (old_valid & (new_key == old_token))
        old_blocked = old_blocked | (old_valid & (new_key == -1))

    miss_mask = new_valid & ~new_found
    avail_mask = old_valid & ~old_found

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    num_slots = tl.sum(avail_mask.to(tl.int32), axis=0)
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    avail_rank_safe = tl.where(avail_mask, avail_rank, 0)
    new_with_offset = new_token.to(tl.int64) + req_offset

    tl.store(slot_scratch_ptr + row_off + avail_rank_safe, cols, mask=avail_mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_with_offset, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(slot_count_ptr + pid, num_slots)


@triton.jit
def sorted_compact_cache_miss_slots_kernel(
        req_ids_ptr,
        old_sorted_ptr,
        old_slots_ptr,
        new_sorted_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
        LOGK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    new_token = tl.load(new_sorted_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_with_offset = new_token + req_offset

    # lower_bound(old_sorted, new_with_offset)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), topk, tl.int32)
    for _ in range(LOGK):
        mid = (lo + hi) // 2
        mid_valid = new_valid & (mid < topk)
        mid_safe = tl.where(mid_valid, mid, 0)
        old_mid = tl.load(old_sorted_ptr + row_off + mid_safe, mask=mid_valid, other=-1).to(tl.int64)
        go_right = old_mid < new_with_offset
        lo = tl.where(new_valid & go_right, mid + 1, lo)
        hi = tl.where(new_valid & ~go_right, mid, hi)
    lo_safe = tl.where(lo < topk, lo, 0)
    old_found = tl.load(old_sorted_ptr + row_off + lo_safe, mask=new_valid & (lo < topk), other=-1).to(tl.int64)
    miss_mask = new_valid & (old_found != new_with_offset)

    old_with_offset = tl.load(old_sorted_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)

    # lower_bound(new_sorted, old_token)
    old_lo = tl.full((BLOCK,), 0, tl.int32)
    old_hi = tl.full((BLOCK,), topk, tl.int32)
    for _ in range(LOGK):
        mid = (old_lo + old_hi) // 2
        mid_valid = old_valid & (mid < topk)
        mid_safe = tl.where(mid_valid, mid, 0)
        new_mid = tl.load(new_sorted_ptr + row_off + mid_safe, mask=mid_valid, other=-1).to(tl.int64)
        go_right = new_mid < old_token
        old_lo = tl.where(old_valid & go_right, mid + 1, old_lo)
        old_hi = tl.where(old_valid & ~go_right, mid, old_hi)
    old_lo_safe = tl.where(old_lo < topk, old_lo, 0)
    new_found = tl.load(new_sorted_ptr + row_off + old_lo_safe, mask=old_valid & (old_lo < topk), other=-1).to(tl.int64)
    avail_mask = old_valid & (new_found != old_token)

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    num_slots = tl.sum(avail_mask.to(tl.int32), axis=0)
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    avail_rank_safe = tl.where(avail_mask, avail_rank, 0)

    old_slot = tl.load(old_slots_ptr + row_off + cols, mask=mask, other=0).to(tl.int32)
    tl.store(slot_scratch_ptr + row_off + avail_rank_safe, old_slot, mask=avail_mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_with_offset, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(slot_count_ptr + pid, num_slots)


@triton.jit
def init_cache_miss_state_step_kernel(
        req_ids_ptr,
        req_state_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        miss_count_ptr,
        free_count_ptr,
        out_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    last_req_id = tl.load(req_state_ptr + pid).to(tl.int64)
    req_changed = req_id != last_req_id

    if bid == 0:
        tl.store(miss_count_ptr + pid, 0)
        tl.store(free_count_ptr + pid, 0)
        tl.store(req_state_ptr + pid, req_id, mask=req_changed)

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = bid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < topk

    old_token = tl.load(slot_to_token_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_valid = req_changed & mask & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)
    tl.store(token_to_slot_ptr + token_map_off + old_token_safe, -1, mask=old_valid)
    tl.store(slot_to_token_ptr + row_off + cols, -1, mask=req_changed & mask)
    tl.store(slot_stamp_ptr + row_off + cols, 0, mask=req_changed & mask)
    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)


@triton.jit
def collect_cache_miss_state_kernel(
        new_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    if pid >= num_reqs:
        return

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = bid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < topk

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)

    slot = tl.load(token_to_slot_ptr + token_map_off + new_token_safe, mask=new_valid, other=-1).to(tl.int32)
    slot_valid = new_valid & (slot >= 0) & (slot < topk)
    slot_safe = tl.where(slot_valid, slot, 0)
    slot_token = tl.load(slot_to_token_ptr + row_off + slot_safe, mask=slot_valid, other=-1).to(tl.int64)
    hit_mask = slot_valid & (slot_token == new_token)

    stamp = tl.load(stamp_ptr).to(tl.int32)
    tl.store(slot_stamp_ptr + row_off + slot_safe, stamp, mask=hit_mask)

    miss_mask = new_valid & (hit_mask == 0)
    miss_pos = tl.atomic_add(miss_count_ptr + pid, miss_mask.to(tl.int32), sem="relaxed")
    tl.store(miss_scratch_ptr + row_off + miss_pos, new_token, mask=miss_mask)


@triton.jit
def apply_cache_miss_state_atomic_kernel(
        out_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        free_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    if pid >= num_reqs:
        return

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = bid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < topk

    stamp = tl.load(stamp_ptr).to(tl.int32)
    num_miss = tl.load(miss_count_ptr + pid).to(tl.int32)

    slot_token = tl.load(slot_to_token_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    slot_stamp = tl.load(slot_stamp_ptr + row_off + cols, mask=mask, other=0).to(tl.int32)
    free_mask = mask & ((slot_token < 0) | ((slot_token >= 0) & (slot_stamp != stamp)))

    old_valid = free_mask & (slot_token >= 0) & (slot_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, slot_token, 0)
    tl.store(token_to_slot_ptr + token_map_off + old_token_safe, -1, mask=old_valid)
    tl.store(slot_to_token_ptr + row_off + cols, -1, mask=free_mask)
    tl.store(slot_stamp_ptr + row_off + cols, 0, mask=free_mask)

    free_pos = tl.atomic_add(free_count_ptr + pid, free_mask.to(tl.int32), sem="relaxed")
    write_mask = free_mask & (free_pos < num_miss)
    new_token = tl.load(miss_scratch_ptr + row_off + free_pos, mask=write_mask, other=-1).to(tl.int64)
    new_valid = write_mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)

    tl.store(slot_to_token_ptr + row_off + cols, new_token, mask=new_valid)
    tl.store(slot_stamp_ptr + row_off + cols, stamp, mask=new_valid)
    tl.store(token_to_slot_ptr + token_map_off + new_token_safe, cols, mask=new_valid)
    tl.store(out_ptr + row_off + cols, new_token.to(tl.int32), mask=new_valid)


@triton.jit
def compact_cache_miss_state_kernel(
        req_ids_ptr,
        new_ptr,
        req_state_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    last_req_id = tl.load(req_state_ptr + pid).to(tl.int64)
    req_changed = req_id != last_req_id

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_slot_token = tl.load(slot_to_token_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_slot_valid = mask & (old_slot_token >= 0) & (old_slot_token < TOKEN_LIMIT)
    old_slot_token_safe = tl.where(old_slot_valid, old_slot_token, 0)
    tl.store(token_to_slot_ptr + token_map_off + old_slot_token_safe, -1, mask=req_changed & old_slot_valid)
    tl.store(slot_to_token_ptr + row_off + cols, -1, mask=req_changed & mask)
    tl.store(slot_stamp_ptr + row_off + cols, 0, mask=req_changed & mask)
    tl.store(req_state_ptr + pid, req_id, mask=req_changed)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)

    slot_lookup = tl.load(token_to_slot_ptr + token_map_off + new_token_safe, mask=new_valid, other=-1).to(tl.int32)
    slot = tl.where(req_changed, -1, slot_lookup)
    slot_valid = new_valid & (slot >= 0) & (slot < topk)
    slot_safe = tl.where(slot_valid, slot, 0)
    slot_token = tl.load(slot_to_token_ptr + row_off + slot_safe, mask=slot_valid, other=-1).to(tl.int64)
    hit_mask = slot_valid & (slot_token == new_token)

    stamp = tl.load(stamp_ptr).to(tl.int32)
    tl.store(slot_stamp_ptr + row_off + slot_safe, stamp, mask=hit_mask)

    miss_mask = new_valid & (hit_mask == 0)

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)

    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_token, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)


@triton.jit
def apply_cache_miss_state_kernel(
        req_ids_ptr,
        out_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)

    stamp = tl.load(stamp_ptr).to(tl.int32)
    num_miss = tl.load(miss_count_ptr + pid).to(tl.int32)

    slot_token = tl.load(slot_to_token_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    slot_stamp = tl.load(slot_stamp_ptr + row_off + cols, mask=mask, other=0).to(tl.int32)
    empty_slot_mask = mask & (slot_token < 0)
    stale_slot_mask = mask & (slot_token >= 0) & (slot_stamp != stamp)
    free_mask = empty_slot_mask | stale_slot_mask
    free_rank = tl.cumsum(free_mask.to(tl.int32), axis=0) - 1
    free_rank_safe = tl.where(free_mask, free_rank, 0)

    old_valid = free_mask & (slot_token >= 0) & (slot_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, slot_token, 0)
    tl.store(token_to_slot_ptr + token_map_off + old_token_safe, -1, mask=old_valid)
    tl.store(slot_to_token_ptr + row_off + cols, -1, mask=free_mask)
    tl.store(slot_stamp_ptr + row_off + cols, 0, mask=free_mask)

    write_mask = free_mask & (free_rank < num_miss)
    new_token = tl.load(miss_scratch_ptr + row_off + free_rank_safe, mask=write_mask, other=-1).to(tl.int64)
    new_valid = write_mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)

    tl.store(slot_to_token_ptr + row_off + cols, new_token, mask=new_valid)
    tl.store(slot_stamp_ptr + row_off + cols, stamp, mask=new_valid)
    tl.store(token_to_slot_ptr + token_map_off + new_token_safe, cols, mask=new_valid)
    tl.store(out_ptr + row_off + cols, new_token.to(tl.int32), mask=new_valid)


@triton.jit
def mark_compact_apply_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        miss_scratch_ptr,
        out_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    new_with_offset = new_token + req_offset

    stamp = tl.load(stamp_ptr).to(tl.int32)
    stamp_vals = tl.full((BLOCK,), 0, tl.int32) + stamp
    tl.store(old_marker_ptr + marker_off + old_token_safe, stamp_vals, mask=old_valid)
    tl.store(new_marker_ptr + marker_off + new_token_safe, stamp_vals, mask=new_valid)

    old_hit = tl.load(old_marker_ptr + marker_off + new_token_safe, mask=new_valid, other=0)
    new_hit = tl.load(new_marker_ptr + marker_off + old_token_safe, mask=old_valid, other=0)

    miss_mask = new_valid & (old_hit != stamp)
    avail_mask = old_valid & (new_hit != stamp)

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    avail_rank_safe = tl.where(avail_mask, avail_rank, 0)

    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_with_offset, mask=miss_mask)

    write_mask = avail_mask & (avail_rank < num_miss)
    miss_with_offset = tl.load(miss_scratch_ptr + row_off + avail_rank_safe, mask=write_mask, other=-1).to(tl.int64)
    miss_token = miss_with_offset - req_offset

    tl.store(old_ptr + row_off + cols, miss_with_offset, mask=write_mask)
    tl.store(out_ptr + row_off + cols, miss_token.to(tl.int32), mask=write_mask)


@triton.jit
def mark_and_compact_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    new_with_offset = new_token + req_offset

    stamp = tl.load(stamp_ptr).to(tl.int32)
    stamp_vals = tl.full((BLOCK,), 0, tl.int32) + stamp
    tl.store(old_marker_ptr + marker_off + old_token_safe, stamp_vals, mask=old_valid)
    tl.store(new_marker_ptr + marker_off + new_token_safe, stamp_vals, mask=new_valid)

    old_hit = tl.load(old_marker_ptr + marker_off + new_token_safe, mask=new_valid, other=0)
    new_hit = tl.load(new_marker_ptr + marker_off + old_token_safe, mask=old_valid, other=0)

    miss_mask = new_valid & (old_hit != stamp)
    avail_mask = old_valid & (new_hit != stamp)

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    num_slots = tl.sum(avail_mask.to(tl.int32), axis=0)
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    avail_rank_safe = tl.where(avail_mask, avail_rank, 0)

    tl.store(slot_scratch_ptr + row_off + avail_rank_safe, cols, mask=avail_mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_with_offset, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(slot_count_ptr + pid, num_slots)


@triton.jit
def mark_cache_tokens_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)
    stamp = tl.load(stamp_ptr).to(tl.int32)
    stamp_vals = tl.full((BLOCK,), 0, tl.int32) + stamp
    tl.store(old_marker_ptr + marker_off + old_token_safe, stamp_vals, mask=old_valid)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    tl.store(new_marker_ptr + marker_off + new_token_safe, stamp_vals, mask=new_valid)


@triton.jit
def compact_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    new_with_offset = new_token + req_offset

    old_hit = tl.load(old_marker_ptr + marker_off + new_token_safe, mask=new_valid, other=0)
    new_hit = tl.load(new_marker_ptr + marker_off + old_token_safe, mask=old_valid, other=0)

    stamp = tl.load(stamp_ptr).to(tl.int32)
    miss_mask = new_valid & (old_hit != stamp)
    avail_mask = old_valid & (new_hit != stamp)

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    num_slots = tl.sum(avail_mask.to(tl.int32), axis=0)
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    avail_rank_safe = tl.where(avail_mask, avail_rank, 0)

    tl.store(slot_scratch_ptr + row_off + avail_rank_safe, cols, mask=avail_mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_with_offset, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(slot_count_ptr + pid, num_slots)


@triton.jit
def apply_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        out_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)

    num_miss = tl.load(miss_count_ptr + pid).to(tl.int32)
    num_slots = tl.load(slot_count_ptr + pid).to(tl.int32)
    update_mask = cols < num_slots
    miss_mask = cols < num_miss
    slots = tl.load(slot_scratch_ptr + row_off + cols, mask=update_mask, other=0).to(tl.int32)
    miss_with_offset = tl.load(miss_scratch_ptr + row_off + cols, mask=miss_mask, other=-1).to(tl.int64)
    miss_token = miss_with_offset - req_offset
    out_mask = miss_mask & update_mask

    tl.store(old_ptr + row_off + slots, miss_with_offset, mask=update_mask)
    tl.store(out_ptr + row_off + slots, miss_token.to(tl.int32), mask=out_mask)


@triton.jit
def get_cache_miss_topk_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        out_ptr,
        num_reqs,
        topk: tl.constexpr,
        BLOCK: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int32)
    req_offset = req_id * 65536
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
    # ---- sub-blocked miss_mask: new not in old ----
    miss_count = tl.zeros((BLOCK,), tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        old_chunk = tl.load(
            old_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int32)
        old_b = tl.broadcast_to(old_chunk[None, :], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_with_offset[:, None], (BLOCK, SUB_BLOCK))
        cmp = new_b == old_b
        miss_count += tl.sum(cmp.to(tl.int32), axis=1)
    miss_mask = (miss_count == 0) & (new_with_offset >= 0)

    # ---- sub-blocked avail_mask: old not in new ----
    avail_count = tl.zeros((BLOCK,), tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        new_chunk = tl.load(
            new_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int32)
        new_chunk_off = tl.where(new_chunk >= 0, new_chunk + req_offset, -1)
        old_b = tl.broadcast_to(old[:, None], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_chunk_off[None, :], (BLOCK, SUB_BLOCK))
        cmp = old_b == new_b
        avail_count += tl.sum(cmp.to(tl.int32), axis=1)
    avail_mask = (avail_count == 0) & (old >= 0)

    # ---- shortage: fill empty slots ----
    num_tokens_to_load = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_available_slot = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    empty_mask = old == -1
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage_slot) & empty_mask
    avail_mask = avail_mask | selected_empty

    # ---- compact: scatter miss values into available slots ----
    miss_vals = tl.where(miss_mask, new_with_offset, 0)
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)

    # Gather-then-scatter: split by SUB_BLOCK chunks of target rank
    # Phase 1 (gather): for each target rank r in [sb_start, sb_start+SUB_BLOCK),
    #            find miss_vals where miss_rank == r
    # Phase 2 (scatter): for each available slot where avail_rank == r,
    #            write the gathered value
    out_with_offset = tl.full((BLOCK,), -1, tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        target_ranks = sb_start + tl.arange(0, SUB_BLOCK)

        # Phase 1: gather - [BLOCK, SUB_BLOCK]
        mr_b = tl.broadcast_to(miss_rank[:, None], (BLOCK, SUB_BLOCK))
        tr_b = tl.broadcast_to(target_ranks[None, :], (BLOCK, SUB_BLOCK))
        mv_b = tl.broadcast_to(miss_vals[:, None], (BLOCK, SUB_BLOCK))
        mm_b = tl.broadcast_to(miss_mask[:, None], (BLOCK, SUB_BLOCK))

        miss_match = (mr_b == tr_b) & mm_b
        gathered = tl.sum(
            tl.where(miss_match, mv_b, tl.zeros((BLOCK, SUB_BLOCK), tl.int32)),
            axis=0,
        )

        # Phase 2: scatter - [BLOCK, SUB_BLOCK]
        ar_b = tl.broadcast_to(avail_rank[:, None], (BLOCK, SUB_BLOCK))
        am_b = tl.broadcast_to(avail_mask[:, None], (BLOCK, SUB_BLOCK))
        valid_rank = tr_b < num_miss

        slot_match = (ar_b == tr_b) & am_b & valid_rank
        result = tl.sum(
            tl.where(
                slot_match,
                gathered[None, :],
                tl.zeros((BLOCK, SUB_BLOCK), tl.int32),
            ),
            axis=1,
        )
        has_match = tl.sum(slot_match.to(tl.int32), axis=1) > 0
        out_with_offset = tl.where(has_match, result, out_with_offset)

    # ---- update old in-place ----
    updated_old = tl.where(avail_mask, out_with_offset, old)
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)

    # ---- remove req offset and store ----
    out = tl.where(out_with_offset >= 0, out_with_offset - req_offset, tl.full((BLOCK,), -1, tl.int32))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)


def get_cache_miss_topk_indices_triton_bitmap(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    token_limit: int = 65536,
    old_marker: torch.Tensor | None = None,
    new_marker: torch.Tensor | None = None,
    slot_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    slot_count: torch.Tensor | None = None,
    stamp_tensor: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    fuse_apply: bool = True,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    if out is None:
        out = torch.empty(
            topk_indices_new.shape,
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if old_marker is None:
        old_marker = torch.zeros(
            (num_reqs, token_limit),
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if new_marker is None:
        new_marker = torch.zeros(
            (num_reqs, token_limit),
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if slot_scratch is None:
        slot_scratch = torch.empty(
            topk_indices_new.shape,
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if miss_scratch is None:
        miss_scratch = torch.empty(
            topk_indices_old.shape,
            dtype=topk_indices_old.dtype,
            device=topk_indices_old.device,
        )
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=topk_indices_new.device)
    if slot_count is None:
        slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=topk_indices_new.device)
    if stamp_tensor is None:
        stamp_tensor = torch.ones((1,), dtype=torch.int32, device=topk_indices_new.device)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    if fuse_apply:
        mark_compact_apply_cache_miss_slots_kernel[grid](
            req_ids_tensor,
            topk_indices_old,
            topk_indices_new,
            old_marker,
            new_marker,
            miss_scratch,
            out,
            num_reqs,
            stamp_tensor,
            topk=topk,
            TOKEN_LIMIT=token_limit,
            BLOCK=BLOCK,
        )
        return out

    mark_and_compact_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_marker,
        new_marker,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        stamp_tensor,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    apply_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        out,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    return out


def get_cache_miss_topk_indices_triton_exact(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    sub_block: int = 4,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty(
        topk_indices_new.shape,
        dtype=torch.int32,
        device=topk_indices_new.device,
    )

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    get_cache_miss_topk_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        out,
        num_reqs,
        topk=topk,
        BLOCK=BLOCK,
        SUB_BLOCK=sub_block,
    )
    return out


def get_cache_miss_topk_indices_triton(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    **kwargs,
):
    return get_cache_miss_topk_indices_triton_hash(
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        **kwargs,
    )


def get_cache_miss_topk_indices_triton_hash(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    token_limit: int = 65536,
    old_marker: torch.Tensor | None = None,
    new_marker: torch.Tensor | None = None,
    stamp_tensor: torch.Tensor | None = None,
    old_hash: torch.Tensor | None = None,
    new_hash: torch.Tensor | None = None,
    slot_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    slot_count: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    hash_capacity: int = 8192,
    probe: int = 16,
    debug_sync: bool = True,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    device = topk_indices_new.device
    if out is None:
        out = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
    if old_hash is None:
        old_hash = torch.empty((num_reqs, hash_capacity), dtype=torch.int32, device=device)
    if new_hash is None:
        new_hash = torch.empty((num_reqs, hash_capacity), dtype=torch.int32, device=device)
    if slot_scratch is None:
        slot_scratch = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
    if miss_scratch is None:
        miss_scratch = torch.empty((num_reqs, topk), dtype=topk_indices_old.dtype, device=device)
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
    if slot_count is None:
        slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)

    topk_block = triton.next_power_of_2(topk)
    clear_block = 256
    clear_grid = (num_reqs, triton.cdiv(hash_capacity, clear_block))
    grid = (num_reqs,)

    clear_cache_miss_hash_kernel[clear_grid](
        old_hash,
        new_hash,
        num_reqs,
        HASH_CAPACITY=hash_capacity,
        BLOCK=clear_block,
    )
    if debug_sync:
        try:
            torch.npu.synchronize()
        except Exception as exc:
            raise RuntimeError("clear_cache_miss_hash_kernel failed") from exc

    mark_cache_tokens_hash_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_hash,
        new_hash,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        HASH_CAPACITY=hash_capacity,
        BLOCK=topk_block,
        PROBE=probe,
    )
    if debug_sync:
        try:
            torch.npu.synchronize()
        except Exception as exc:
            raise RuntimeError("mark_cache_tokens_hash_kernel failed") from exc

    compact_cache_miss_slots_hash_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_hash,
        new_hash,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        HASH_CAPACITY=hash_capacity,
        BLOCK=topk_block,
        PROBE=probe,
    )
    if debug_sync:
        try:
            torch.npu.synchronize()
        except Exception as exc:
            raise RuntimeError("compact_cache_miss_slots_hash_kernel failed") from exc

    apply_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        out,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=topk_block,
    )
    if debug_sync:
        try:
            torch.npu.synchronize()
        except Exception as exc:
            raise RuntimeError("apply_cache_miss_slots_kernel failed after hash compact") from exc

    return out


def get_cache_miss_topk_indices_triton_sorted(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new_sorted: torch.Tensor,
    topk_indices_old_sorted: torch.Tensor,
    topk_indices_old_slots: torch.Tensor,
    token_limit: int = 65536,
    slot_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    slot_count: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    num_reqs, topk = topk_indices_new_sorted.shape
    assert topk == topk_indices_old.shape[1]
    assert topk == topk_indices_old_sorted.shape[1]
    assert topk == topk_indices_old_slots.shape[1]

    device = topk_indices_new_sorted.device
    if out is None:
        out = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
    if slot_scratch is None:
        slot_scratch = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
    if miss_scratch is None:
        miss_scratch = torch.empty((num_reqs, topk), dtype=topk_indices_old.dtype, device=device)
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
    if slot_count is None:
        slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)
    LOGK = topk.bit_length()

    sorted_compact_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old_sorted,
        topk_indices_old_slots,
        topk_indices_new_sorted,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
        LOGK=LOGK,
    )

    apply_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        out,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    return out


def get_cache_miss_topk_indices_triton_state(
    req_ids_tensor: torch.Tensor,
    topk_indices_new: torch.Tensor,
    token_limit: int = 65536,
    use_atomic: bool = False,
    req_state: torch.Tensor | None = None,
    token_to_slot: torch.Tensor | None = None,
    slot_to_token: torch.Tensor | None = None,
    slot_stamp: torch.Tensor | None = None,
    free_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    free_count: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    stamp_tensor: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    num_reqs, topk = topk_indices_new.shape
    device = topk_indices_new.device

    if req_state is None:
        req_state = torch.full((num_reqs,), -1, dtype=req_ids_tensor.dtype, device=device)
    if token_to_slot is None:
        token_to_slot = torch.full((num_reqs, token_limit), -1, dtype=torch.int32, device=device)
    if slot_to_token is None:
        slot_to_token = torch.full((num_reqs, topk), -1, dtype=torch.int64, device=device)
    if slot_stamp is None:
        slot_stamp = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)
    if miss_scratch is None:
        miss_scratch = torch.empty((num_reqs, topk), dtype=torch.int64, device=device)
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
    if stamp_tensor is None:
        stamp_tensor = torch.ones((1,), dtype=torch.int32, device=device)
    if out is None:
        out = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)

    if use_atomic:
        # Legacy 3-kernel atomic path
        if free_scratch is None:
            free_scratch = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
        if free_count is None:
            free_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)

        BLOCK = 128
        grid = (num_reqs, triton.cdiv(topk, BLOCK))

        init_cache_miss_state_step_kernel[grid](
            req_ids_tensor,
            req_state,
            token_to_slot,
            slot_to_token,
            slot_stamp,
            miss_count,
            free_count,
            out,
            num_reqs,
            topk=topk,
            TOKEN_LIMIT=token_limit,
            BLOCK=BLOCK,
        )

        collect_cache_miss_state_kernel[grid](
            topk_indices_new,
            token_to_slot,
            slot_to_token,
            slot_stamp,
            miss_scratch,
            miss_count,
            num_reqs,
            stamp_tensor,
            topk=topk,
            TOKEN_LIMIT=token_limit,
            BLOCK=BLOCK,
        )

        apply_cache_miss_state_atomic_kernel[grid](
            out,
            token_to_slot,
            slot_to_token,
            slot_stamp,
            miss_scratch,
            miss_count,
            free_count,
            num_reqs,
            stamp_tensor,
            topk=topk,
            TOKEN_LIMIT=token_limit,
            BLOCK=BLOCK,
        )
    else:
        # Non-atomic 2-kernel path: compact + apply
        # One program per request processes all topk slots at once.
        # compact handles req_id change detection, old state cleanup,
        # token_to_slot lookup, hit/miss classification, and miss compaction
        # using tl.cumsum (no atomics).
        # apply finds free slots (empty + stale), pairs them with misses,
        # updates all state arrays, and produces the slot-aligned output.
        BLOCK = triton.next_power_of_2(topk)
        grid = (num_reqs,)

        compact_cache_miss_state_kernel[grid](
            req_ids_tensor,
            topk_indices_new,
            req_state,
            token_to_slot,
            slot_to_token,
            slot_stamp,
            miss_scratch,
            miss_count,
            num_reqs,
            stamp_tensor,
            topk=topk,
            TOKEN_LIMIT=token_limit,
            BLOCK=BLOCK,
        )

        apply_cache_miss_state_kernel[grid](
            req_ids_tensor,
            out,
            token_to_slot,
            slot_to_token,
            slot_stamp,
            miss_scratch,
            miss_count,
            num_reqs,
            stamp_tensor,
            topk=topk,
            TOKEN_LIMIT=token_limit,
            BLOCK=BLOCK,
        )

    return out


class CacheMissTopKScratch:

    def __init__(self):
        self.token_limit = None
        self.device = None
        self.history_dtype = None
        self.marker_shape = (0, 0)
        self.scratch_shape = (0, 0)
        self.stamp = 0
        self.old_marker = None
        self.new_marker = None
        self.slot_scratch = None
        self.miss_scratch = None
        self.miss_count = None
        self.slot_count = None
        self.stamp_tensor = None
        self.out = None
        self.old_hash = None
        self.new_hash = None
        self.hash_capacity = None

    def prepare(
        self,
        num_reqs: int,
        topk: int,
        device,
        token_limit: int = 65536,
        history_dtype: torch.dtype = torch.int64,
        hash_capacity: int = 8192,
    ):
        marker_shape = (num_reqs, token_limit)
        scratch_shape = (num_reqs, topk)
        needs_alloc = (
            self.token_limit != token_limit
            or self.device != device
            or self.history_dtype != history_dtype
            or self.hash_capacity != hash_capacity
            or self.marker_shape[0] < num_reqs
            or self.scratch_shape[0] < num_reqs
            or self.scratch_shape[1] != topk
        )

        if needs_alloc:
            self.old_marker = torch.zeros(marker_shape, dtype=torch.int32, device=device)
            self.new_marker = torch.zeros(marker_shape, dtype=torch.int32, device=device)
            self.slot_scratch = torch.empty(scratch_shape, dtype=torch.int32, device=device)
            self.miss_scratch = torch.empty(scratch_shape, dtype=history_dtype, device=device)
            self.miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.stamp_tensor = torch.empty((1,), dtype=torch.int32, device=device)
            self.out = torch.empty(scratch_shape, dtype=torch.int32, device=device)
            self.old_hash = torch.empty((num_reqs, hash_capacity), dtype=torch.int32, device=device)
            self.new_hash = torch.empty((num_reqs, hash_capacity), dtype=torch.int32, device=device)
            self.token_limit = token_limit
            self.device = device
            self.history_dtype = history_dtype
            self.hash_capacity = hash_capacity
            self.marker_shape = marker_shape
            self.scratch_shape = scratch_shape
            self.stamp = 0

        self.stamp += 1
        if self.stamp >= 2_000_000_000:
            self.old_marker.zero_()
            self.new_marker.zero_()
            self.stamp = 1
        self.stamp_tensor.fill_(self.stamp)

        return {
            "token_limit": token_limit,
            "stamp_tensor": self.stamp_tensor,
            "old_marker": self.old_marker[:num_reqs],
            "new_marker": self.new_marker[:num_reqs],
            "slot_scratch": self.slot_scratch[:num_reqs, :topk],
            "miss_scratch": self.miss_scratch[:num_reqs, :topk],
            "miss_count": self.miss_count[:num_reqs],
            "slot_count": self.slot_count[:num_reqs],
            "out": self.out[:num_reqs, :topk],
            "old_hash": self.old_hash[:num_reqs],
            "new_hash": self.new_hash[:num_reqs],
            "hash_capacity": hash_capacity,
        }


class CacheMissTopKState:

    def __init__(self):
        self.token_limit = None
        self.device = None
        self.req_dtype = None
        self.shape = (0, 0)
        self.stamp = 0
        self.req_state = None
        self.token_to_slot = None
        self.slot_to_token = None
        self.slot_stamp = None
        self.free_scratch = None
        self.miss_scratch = None
        self.free_count = None
        self.miss_count = None
        self.stamp_tensor = None
        self.out = None

    def prepare(
        self,
        num_reqs: int,
        topk: int,
        device,
        req_dtype: torch.dtype,
        token_limit: int = 65536,
    ):
        needs_alloc = (
            self.token_limit != token_limit
            or self.device != device
            or self.req_dtype != req_dtype
            or self.shape[0] < num_reqs
            or self.shape[1] != topk
        )

        if needs_alloc:
            self.req_state = torch.full((num_reqs,), -1, dtype=req_dtype, device=device)
            self.token_to_slot = torch.full((num_reqs, token_limit), -1, dtype=torch.int32, device=device)
            self.slot_to_token = torch.full((num_reqs, topk), -1, dtype=torch.int64, device=device)
            self.slot_stamp = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)
            self.free_scratch = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
            self.miss_scratch = torch.empty((num_reqs, topk), dtype=torch.int64, device=device)
            self.free_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.stamp_tensor = torch.empty((1,), dtype=torch.int32, device=device)
            self.out = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
            self.token_limit = token_limit
            self.device = device
            self.req_dtype = req_dtype
            self.shape = (num_reqs, topk)
            self.stamp = 0

        self.stamp += 1
        if self.stamp >= 2_000_000_000:
            self.slot_stamp.zero_()
            self.stamp = 1
        self.stamp_tensor.fill_(self.stamp)

        return {
            "token_limit": token_limit,
            "req_state": self.req_state[:num_reqs],
            "token_to_slot": self.token_to_slot[:num_reqs],
            "slot_to_token": self.slot_to_token[:num_reqs, :topk],
            "slot_stamp": self.slot_stamp[:num_reqs, :topk],
            "free_scratch": self.free_scratch[:num_reqs, :topk],
            "miss_scratch": self.miss_scratch[:num_reqs, :topk],
            "free_count": self.free_count[:num_reqs],
            "miss_count": self.miss_count[:num_reqs],
            "stamp_tensor": self.stamp_tensor,
            "out": self.out[:num_reqs, :topk],
        }
