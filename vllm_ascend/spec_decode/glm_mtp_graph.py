# SPDX-License-Identifier: Apache-2.0
"""Fixed-address inputs and captured preprocessing for three GLM MTP steps.

Host staging and DP consensus stay outside the graph. Padding, position/slot
updates and RoPE gathering run at the beginning of the merged draft graph.
"""

import torch

from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla

_PADDING_SLOT_ID = -1


class GLMMTPGraphBuffers:
    def __init__(self, proposer, metadata_cls, attn_state):
        self.proposer = proposer
        self.metadata_cls = metadata_cls
        self.attn_state = attn_state
        self.metadata: dict[int, list[dict]] = {}

    def _build_metadata(self, num_input_tokens):
        p = self.proposer
        graph_batch_size = num_input_tokens // p.decode_threshold
        table = p.runner.input_batch.block_table[p.kv_cache_gid].get_device_tensor()
        block_table = torch.zeros((num_input_tokens, table.shape[1]), dtype=table.dtype, device=table.device)
        positions = torch.zeros_like(p.positions[:num_input_tokens])
        builder = p.draft_attn_groups[0].get_metadata_builder()
        result = []
        for step in range(p.num_speculative_tokens):
            query_len = p.decode_threshold if step == 0 else 1
            num_reqs = graph_batch_size if step == 0 else num_input_tokens
            query_cpu = torch.arange(num_reqs + 1, dtype=torch.int32) * query_len
            # Own storage per graph size: eager builders and other graph keys
            # must not overwrite the constant query layout or dynamic buffers.
            seq_lens = torch.zeros_like(p.seq_lens_group[step][:num_reqs])
            cpu_lens = torch.zeros(num_reqs, dtype=torch.int32)
            metadata = self.metadata_cls(
                query_start_loc=query_cpu.to(device=p.device),
                query_start_loc_cpu=query_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=cpu_lens,
                _seq_lens_cpu=cpu_lens,
                num_reqs=num_reqs,
                num_actual_tokens=num_input_tokens,
                num_input_tokens=num_input_tokens,
                max_query_len=query_len,
                max_seq_len=p.max_model_len,
                block_table_tensor=block_table[:num_reqs],
                slot_mapping=torch.full_like(p.slot_mapping_group[step][:num_input_tokens], _PADDING_SLOT_ID),
                positions=positions,
                attn_state=self.attn_state,
                decode_token_per_req=query_len,
                is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
            )
            if step == 0:
                attn = builder.build(0, metadata, p.runner.get_model())
            else:
                attn = builder.build_for_drafting(metadata, step)
            assert attn.dcp_context is None and attn.dsa_cp_context is None
            # The builder's RoPE and optional C8 grouping scratch can be shared
            # with target/eager paths. Keep private outputs for this graph key.
            for name in ("cos", "sin", "group_len", "group_key_idx", "group_key_cache_idx"):
                tensor = getattr(attn, name, None)
                if tensor is not None:
                    setattr(attn, name, tensor.clone())
            result.append(dict.fromkeys(p.attn_layer_names, attn))
        return result

    def prepare(self, common, num_input_tokens: int, batch_size: int, token_indices_to_sample):
        """Stage changing inputs; no RoPE/slot computation on the replay path."""
        p = self.proposer
        assert num_input_tokens % p.decode_threshold == 0
        graph_batch_size = num_input_tokens // p.decode_threshold
        assert 0 <= batch_size <= graph_batch_size
        num_tokens = batch_size * p.decode_threshold
        assert common is None or common.num_actual_tokens == num_tokens
        if num_input_tokens not in self.metadata:
            self.metadata[num_input_tokens] = self._build_metadata(num_input_tokens)
        result = self.metadata[num_input_tokens]
        first = result[0][p.attn_layer_names[0]]
        # seq_lens == 0 identifies inactive rows inside the captured graph.
        if batch_size < graph_batch_size:
            first.seq_lens[batch_size:].zero_()
        first.seq_lens_cpu.zero_()
        if batch_size:
            p.token_indices_to_sample[:batch_size].copy_(token_indices_to_sample)
            first.seq_lens[:batch_size].copy_(common.seq_lens[:batch_size])
            first.slot_mapping[:num_tokens].copy_(common.slot_mapping[:num_tokens])
            first.block_table[:batch_size].copy_(common.block_table_tensor[:batch_size])
            source_cpu = common._seq_lens_cpu if common._seq_lens_cpu is not None else common.seq_lens_cpu
            assert source_cpu is not None  # Never add a device->host sync here.
            first.seq_lens_cpu[:batch_size].copy_(source_cpu[:batch_size])
        # Host consumers still need current lengths, in stable CPU storage.
        cpu_lens = first.seq_lens_cpu
        for step in range(1, p.num_speculative_tokens):
            next_cpu = result[step][p.attn_layer_names[0]].seq_lens_cpu
            next_cpu.zero_()
            torch.add(cpu_lens[:batch_size], 1, out=next_cpu[:batch_size])
            next_cpu.masked_fill_(next_cpu > p.max_model_len, 1)
            cpu_lens = next_cpu
        return result

    def preprocess(self, num_input_tokens, metadata):
        """Pure device work, captured once with fixed shapes (including idle DP)."""
        p = self.proposer
        graph_batch_size = num_input_tokens // p.decode_threshold
        first = metadata[0][p.attn_layer_names[0]]
        active = first.seq_lens > 0
        token_active = active.repeat_interleave(p.decode_threshold)
        p.input_ids[:num_input_tokens].masked_fill_(~token_active, 0)
        p.positions[:num_input_tokens].masked_fill_(~token_active, 0)
        p.hidden_states[:num_input_tokens].masked_fill_(~token_active[:, None], 0)
        indices = p.token_indices_to_sample[:graph_batch_size]
        indices.masked_fill_(~active, 0)
        accepted_positions = torch.index_select(p.positions[:num_input_tokens], 0, indices.long())
        first.block_table.masked_fill_(~active[:, None], 0)
        first.slot_mapping.masked_fill_(~token_active, _PADDING_SLOT_ID)

        prev_seq_lens = first.seq_lens
        for step, per_layer in enumerate(metadata):
            attn = per_layer[p.attn_layer_names[0]]
            step_positions = p.positions[:num_input_tokens]
            if step:
                next_lens = prev_seq_lens[:graph_batch_size] + 1
                next_lens.masked_fill_(next_lens > p.max_model_len, 1)
                attn.seq_lens[:graph_batch_size].copy_(next_lens.masked_fill(~active, 0))
                next_positions = accepted_positions + step
                valid = active & (next_positions < p.max_model_len)
                clamped = next_positions.masked_fill(~valid, 0)
                step_positions = step_positions.clone()
                step_positions[:graph_batch_size].copy_(clamped)
                blocks = first.block_table.gather(1, (clamped // p.block_size).long().view(-1, 1))
                slots = blocks.view(-1) * p.block_size + clamped % p.block_size
                attn.slot_mapping[:graph_batch_size].copy_(slots.masked_fill(~valid, _PADDING_SLOT_ID))
            get_cos_and_sin_mla(step_positions.long(), out=(attn.cos, attn.sin))
            if getattr(attn, "group_len", None) is not None:
                torch.ops._C_ascend.store_kv_block_metadata(
                    attn.slot_mapping, attn.group_len, attn.group_key_idx, attn.group_key_cache_idx, attn.block_size
                )
            prev_seq_lens = attn.seq_lens
