# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, OffloadMLAAttentionSpec


class CompressAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: "AscendMLAAttentionSpec", block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        # assert isinstance(self.kv_cache_spec, (CompressAttentionSpec, C4IndexerSpec))

        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio

        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_total_computed_tokens //= self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        num_tokens //= self.compress_ratio
        ## TODO: check spec decode
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        alignment_tokens: int | None = None,
    ) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
            retention_interval: Prefix-cache retention interval.
            alignment_tokens: Cache-hit alignment passed by hybrid KV cache
                coordinators. Compressed attention caches logical blocks, so no
                extra block mask is needed here.
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // (self.block_size * self.compress_ratio)

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size * self.compress_ratio,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        self.num_cached_block[request.request_id] = num_full_blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        drop_eagle_block: bool = False,
    ) -> tuple[list[KVCacheBlock], ...]:
        eagle_drop = drop_eagle_block
        # assert isinstance(
        #     kv_cache_spec, Compress4AttentionSpec | Compress128AttentionSpec | C4IndexerSpec
        # ), (
        #     "CompressAttentionManager can only be used for compressor attention groups"
        # )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        logical_block_size = block_size * kv_cache_spec.compress_ratio
        logical_block_hashes = BlockHashListWithBlockSize(block_hashes, block_size, logical_block_size)
        max_num_blocks = max_length // logical_block_size
        for block_hash in itertools.islice(logical_block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if eagle_drop and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()

        while (
            logical_block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * logical_block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """Build the per-spec KV cache manager.

    For DSv4 / DSA path (``MLAAttentionSpec`` with ``compress_ratio>1``), align
    the runtime admission gate with the startup pool-sizing bound the same way
    vLLM PR #40946 does for ``SlidingWindowSpec`` / ``ChunkedLocalAttentionSpec``.
    Without this cap, an admitted request can demand more blocks than the pool
    was sized to back, and ``allocate_slots`` silently returns ``None`` from
    the ``full_sequence_must_fit`` branch, leaving long-input requests stuck
    in the waiting queue (see vLLM issue #40863, observed on DSv4 + MTP with
    cc>=1 and prompt>=32K).

    The compressed-MLA peak per request is bounded by
    ``cdiv(max_model_len // compress_ratio, block_size)`` (it does not shrink
    via recycling like SWA, but neither does it ever exceed this). Capping at
    this value matches the pool sizer and makes admission consistent with the
    block budget actually held.
    """
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # type: ignore[import-not-found]

    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, f"No KV cache manager registered for {type(kv_cache_spec).__name__}"
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            # Compressed-MLA peak in blocks: ceil(max_model_len/compress/block).
            compress_ratio = kv_cache_spec.compress_ratio
            block_size = kv_cache_spec.block_size
            max_compressed_tokens = max_model_len // compress_ratio
            kwargs["max_admission_blocks_per_request"] = cdiv(max_compressed_tokens, block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        # Replicate the upstream PR #40946 cap setting for recycling specs.
        # We override the vLLM factory above, so the upstream block that does
        # this lives in dead code (never reached); without re-applying it here
        # SlidingWindowMLASpec / ChunkedLocalAttentionSpec groups have no cap
        # and ``full_sequence_must_fit`` admission reserves the full
        # ``max_model_len`` worth of blocks per request, exhausting the pool
        # at cc>=2 on DSv4 (see vLLM issue #40863).
        if max_num_batched_tokens is not None and max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


class OffloadMLAAttentionManager(FullAttentionManager):
    """
    SFA kv offload kv cache manager,
    free offloaded blocks before allocating new blocks.
    """
    def __init__(self, kv_cache_spec: "OffloadMLAAttentionSpec", **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.req_to_offloaded_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        self.req_to_num_allocated_tokens: defaultdict[str, int] = defaultdict(int)
        self.decode_threshold: int | None = None
        # Req ids whose main-MLA prefix is remote-prefilled (PD): the prefix
        # lives in the CPU pool, was never in HBM, and is null-padded at the
        # FRONT of req_to_blocks. Real decode-generated blocks ARE freed (to
        # block_pool) once the connector has copied them HBM->CPU -- freed
        # in-place as null_block (NOT popped), so the null prefix and the
        # block-table row length stay intact. req_to_real_free_cursor[rid] is
        # the index in req_to_blocks of the next real block to free (initialized
        # to N = null-prefix length at admission, advances by 1 per free).
        self.req_is_remote_prefilled: set[str] = set()
        self.req_to_real_free_cursor: defaultdict[str, int] = defaultdict(int)

    def free(self, request_id: str) -> None:
        """Override: drop the per-req offload bookkeeping the base free() does
        not know about, then defer to the base to free the real HBM blocks
        (block_pool.free_blocks already skips null_block entries, so the
        null-padded prefix of a remote-prefilled req is not double-freed).
        Without this the three dicts/set leak across the session and a recycled
        request_id would inherit a stale remote-prefilled flag.
        """
        self.req_is_remote_prefilled.discard(request_id)
        self.req_to_offloaded_blocks.pop(request_id, None)
        self.req_to_num_allocated_tokens.pop(request_id, None)
        self.req_to_real_free_cursor.pop(request_id, None)
        super().free(request_id)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.

        Returns:
            The number of blocks to allocate.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
        num_req_offloaded_blocks = len(self.req_to_offloaded_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            # PD remote-prefilled: freed decode blocks are replaced in-place with
            # null_block (NOT popped), so they still count in num_req_blocks;
            # subtracting num_req_offloaded_blocks here would double-count them.
            if request_id in self.req_is_remote_prefilled:
                return max(num_required_blocks - num_req_blocks, 0)
            return max(num_required_blocks - num_req_blocks - num_req_offloaded_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks + num_req_offloaded_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size

        # PD remote-prefill: the external-computed prefix lives in the CPU pool,
        # not HBM. Reserve HBM only for the resident tail (new tokens + the
        # partial block). external = total - local; floor so a partial last
        # block stays in HBM (decode writes into it). For non-PD (local prefill)
        # the connector reports ext=0, so this subtracts nothing -- behavior is
        # unchanged for the existing offload path.
        num_local_computed_tokens = num_local_computed_blocks * self.block_size
        num_external_tokens = max(total_computed_tokens - num_local_computed_tokens, 0)
        num_external_blocks = num_external_tokens // self.block_size

        # Capacity guard: the null-padded prefix occupies [0:num_external_blocks]
        # of the block-table row; resident blocks need the suffix beyond that.
        # If the offloaded prefix fills the whole token span there is no room for
        # any resident (decode) block -> the row would be all nulls and attention
        # would read garbage. There is no post-commit restore to save us now, so
        # surface this loudly. (The block-table row WIDTH is a separate capacity
        # constraint sized by max_model_len; this check covers the degenerate
        # span case.)
        if num_external_blocks >= num_required_blocks:
            logger.warning(
                "OffloadMLA req %s: offloaded prefix (%d blocks) fills the "
                "entire token span (%d required) -- no room for resident KV; "
                "check prompt length vs block_size.",
                request_id,
                num_external_blocks,
                num_required_blocks,
            )

        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks) - num_external_blocks,
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Override: for external (PD remote-prefilled) tokens, NULL-PAD the
        block table instead of allocating real HBM blocks.

        The external prefix is in the CPU pool (pulled from P), so it must not
        consume HBM. Null-padding keeps the block table the right length so
        attention's block_table[index] stays in range; the offloaded positions
        are masked out by the ``num_offloaded_blocks`` attention threshold. The
        base class instead calls ``block_pool.get_new_blocks`` here, which is
        what over-materializes HBM for PD reqs.

        The partial remainder (when ``num_external_computed_tokens`` is not a
        whole multiple of ``block_size``) is left for ``allocate_new_blocks`` to
        allocate as the resident decode block (floor, not cdiv).
        """
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            return

        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the local computed blocks.
        req_blocks.extend(new_computed_blocks)

        if num_external_computed_tokens > 0:
            # PD remote-prefill: null-pad the external (CPU-pool) prefix. Do NOT
            # call block_pool.get_new_blocks -- those HBM blocks are never needed
            # (the prefix is read from CPU via the LRU resident load path).
            # COUPLING: num_external_blocks here MUST equal the worker's
            # num_offloaded_blocks (= prompt_len // _main_block_size) which the
            # SFA kernel uses as its mask threshold. Both floor-divide by the
            # main-MLA block_size. They match when DCP*PCP == 1 (both derive
            # from the main group spec); under DCP/PCP > 1 vLLM core scales
            # self.block_size (SingleTypeKVCacheManager.__init__) while
            # _main_block_size stays raw, so the two diverge and this
            # null-pad/mask coupling is NOT supported. Log it so a hardware run
            # can cross-check manager null-pad width vs kernel mask width.
            self.req_is_remote_prefilled.add(request_id)
            num_external_blocks = num_external_computed_tokens // self.block_size
            logger.info(
                "OffloadMLA null-pad req %s: %d external blocks (block_size=%d, "
                "external_tokens=%d)",
                request_id,
                num_external_blocks,
                self.block_size,
                num_external_computed_tokens,
            )
            req_blocks.extend([self._null_block] * num_external_blocks)
            # Cursor for in-place decode-block freeing (see allocate_new_blocks):
            # starts at N = the index of the first real block (right after the
            # null-padded prefix) and advances by 1 per freed decode block.
            self.req_to_real_free_cursor[request_id] = num_external_blocks

        self.num_cached_block[request_id] = len(req_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        First free the already offloaded blocks to move space,
        then allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        if self.decode_threshold is None:
            # whether current request is prefill or decode,
            # decode_threshold = 1 + spec_decode_size,
            # can't touch vllm config here, have to get it during scheduling.
            self.decode_threshold = num_tokens - num_tokens_main_model + 1

        req_blocks = self.req_to_blocks[request_id]
        req_freed_blocks = self.req_to_offloaded_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        is_remote_prefilled = request_id in self.req_is_remote_prefilled

        # free old full blocks (which should be already offloaded to CPU by the
        # connector in a prior step -- one-step slack, same convention for both
        # paths).
        to_free_blocks: list[KVCacheBlock] = []
        if is_remote_prefilled:
            # PD: free real decode blocks IN-PLACE (replace with null_block, do
            # NOT pop) so the null-padded prefix at [0:N] and the row length
            # stay intact (the block-table length and the attention
            # num_offloaded_blocks mask depend on them). The cursor points at
            # the oldest unfree real block (starts at N at admission).
            num_allocated_tokens = self.req_to_num_allocated_tokens[request_id]
            num_offloaded_total = num_allocated_tokens // self.block_size
            cursor = self.req_to_real_free_cursor[request_id]
            num_to_free_real = max(
                min(num_offloaded_total - cursor, len(req_blocks) - cursor), 0
            )
            for _ in range(num_to_free_real):
                blk = req_blocks[cursor]
                assert blk is not self._null_block, (
                    f"req {request_id}: PD free cursor hit null_block at index "
                    f"{cursor} (cursor desync)"
                )
                req_blocks[cursor] = self._null_block
                req_freed_blocks.append(blk)
                to_free_blocks.append(blk)
                cursor += 1
            self.req_to_real_free_cursor[request_id] = cursor
        else:
            num_allocated_tokens = self.req_to_num_allocated_tokens[request_id]
            num_new_tokens_main_model = num_tokens_main_model - num_allocated_tokens
            if num_new_tokens_main_model > self.decode_threshold:
                # (chunk) prefill case, should not release any blocks
                num_to_free_blocks = 0
            else:
                # only offload & free after (chunk) prefill is done
                num_offloaded_blocks = num_allocated_tokens // self.block_size
                num_freed_blocks = len(req_freed_blocks)
                num_to_free_blocks = num_offloaded_blocks - num_freed_blocks
                # Defensive: never pop more real blocks than req_blocks actually
                # holds, and never pop the null sentinel.
                num_to_free_blocks = max(min(num_to_free_blocks, len(req_blocks)), 0)
            for _ in range(num_to_free_blocks):
                to_free_block = req_blocks.pop(0)
                assert to_free_block is not self._null_block, (
                    f"req {request_id}: tried to free the null block "
                    f"(offload accounting is inconsistent)"
                )
                req_freed_blocks.append(to_free_block)
                to_free_blocks.append(to_free_block)
        if to_free_blocks:
            self.block_pool.free_blocks(to_free_blocks)
            logger.info(f'>>>>> kv cache manager, req {request_id} free {len(to_free_blocks)} offloaded blocks: {[block.block_id for block in to_free_blocks]}')

        # allocate new blocks. PD remote-prefilled: freed decode blocks stay in
        # req_blocks as null_block (in-place, not popped), so len(req_blocks)
        # already accounts for them -- do NOT subtract len(req_freed_blocks)
        # (would double-count and starve the resident tail).
        if is_remote_prefilled:
            num_new_blocks = num_required_blocks - len(req_blocks)
        else:
            num_new_blocks = num_required_blocks - len(req_blocks) - len(req_freed_blocks)
        # req_to_num_allocated_tokens records the FULL logical length (prefix +
        # decode). The PD free branch reads it as num_offloaded_total and offsets
        # by the cursor (= N at admission) to count only real decode blocks that
        # should be freed.
        self.req_to_num_allocated_tokens[request_id] = num_tokens_main_model
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0
