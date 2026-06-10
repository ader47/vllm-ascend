from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu
from vllm.distributed import get_dcp_group, get_decode_context_model_parallel_world_size, get_pcp_group
from vllm.logger import logger


@dataclass
class AscendPCPMetadata:
    """
    Metadata for Prefill Context Parallelism (PCP) on Ascend devices.

    Stores index tensors and sequence lengths for routing attention
    computations across PCP ranks during long sequence processing.
    """

    q_head_idx: torch.Tensor = None
    q_tail_idx: torch.Tensor = None
    kv_with_q_head_nomask_idx: torch.Tensor = None
    kv_with_q_head_mask_idx: torch.Tensor = None
    kv_with_q_tail_nomask_idx: torch.Tensor = None
    kv_with_q_tail_mask_idx: torch.Tensor = None
    kv_tail_proj_idx: torch.Tensor = None
    kv_with_q_head_attn_idx_in_tail: torch.Tensor = None
    kv_with_q_tail_attn_idx_in_tail: torch.Tensor = None
    attn_mask_seqlens: torch.Tensor = None
    head_attn_nomask_seqlens: torch.Tensor = None
    tail_attn_nomask_seqlens: torch.Tensor = None
    head_actual_seq_lengths_kv: list[int] | None = None
    tail_actual_seq_lengths_kv: list[int] | None = None
    q_full_idx: torch.Tensor = None
    pcp_use_hybrid_attn: bool = False
    pcp_unpad_mask: torch.Tensor = None
    pcp_allgather_restore_idx: list[int] | None = None
    pcp_fa_query_idx: torch.Tensor = None
    pcp_padded_tokens_fla: int = 0
    pcp_enter_fa_restore_idx: torch.Tensor = None
    block_table_cp: torch.Tensor = None
    valid_block_ids: torch.Tensor = None
    prefill_q_cum_seqlens: torch.Tensor = None
    max_num_tokens_across_pcp: int = 0
    total_num_scheduled_tokens: int = 0
    block_arange: torch.Tensor = None


@dataclass
class CPChunkedContextMetadata:
    """
    Metadata for chunked context handling in Context Parallelism (CP).

    Extends chunked prefill with per-rank chunk information for PCP/DCP.
    """

    # For handling chunked prefill
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    workspace: torch.Tensor
    chunk_seq_lens: torch.Tensor
    chunk_seq_lens_npu: torch.Tensor
    chunk_actual_seq_lengths_kv_list: list[list[int]]
    # for mla DCP & PCP
    padded_chunk_seq_lens_npu: torch.Tensor = None
    padded_local_chunk_seq_lens: list[list[int]] | None = None
    local_context_lens_allranks: list[list[int]] | None = None
    padded_local_cu_seq_lens: torch.Tensor = None
    cu_seq_lens_lst: list[list[int]] | None = None
    chunk_size: int | None = None


@dataclass
class AscendMetadataForPrefill:
    """Prefill-specific metadata for Ascend attention with Context Parallelism."""

    @dataclass
    class ChunkedContextMetadata:
        """Metadata for chunked context processing within prefill phase."""

        actual_chunk_seq_lengths: torch.Tensor
        actual_seq_lengths_kv: torch.Tensor
        starts: torch.Tensor
        chunk_seq_mask_filtered_indices: torch.Tensor
        chunked_req_mask: list[bool] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        cp_kv_recover_idx_for_chunk: list[int] | None = None
        kv_inverse_idx_for_chunk: list[int] | None = None
        local_total_toks: int | None = None

    """ Prefill Specific Metadata for Ascend"""
    pcp_metadata: AscendPCPMetadata | None = None
    pcp_exit_fa_scatter_idx: torch.Tensor | None = None
    chunked_context: ChunkedContextMetadata | None = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: torch.Tensor = None


@dataclass
class AscendMetadataForDecode:
    """Decode-specific metadata for Ascend attention with Context Parallelism."""

    num_computed_tokens_of_pcp_dcp: list[list[list[int]]] | None = None
    block_tables: torch.Tensor = None
    dcp_mtp_attn_mask: torch.Tensor = None


def _process_attn_out_lse(attn_output: torch.Tensor, softmax_lse: torch.Tensor) -> torch.Tensor:
    pcp_size = get_pcp_group().world_size
    dcp_size = get_decode_context_model_parallel_world_size()
    dcp_group = get_dcp_group().device_group if dcp_size > 1 else None

    # [DCP_DEBUG] Log shapes before processing
    logger.info(
        "[DCP_DEBUG] _process_attn_out_lse BEFORE: "
        "pcp_size=%d, dcp_size=%d, "
        "attn_output.shape=%s, softmax_lse.shape=%s, "
        "attn_output stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, "
        "softmax_lse stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        pcp_size,
        dcp_size,
        str(attn_output.shape),
        str(softmax_lse.shape),
        attn_output.float().mean().item(),
        attn_output.float().std().item(),
        attn_output.float().min().item(),
        attn_output.float().max().item(),
        softmax_lse.float().mean().item(),
        softmax_lse.float().std().item(),
        softmax_lse.float().min().item(),
        softmax_lse.float().max().item(),
    )

    softmax_lse = softmax_lse.to(torch.float32)
    attn_output = attn_output.to(torch.float32)
    # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
    attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
    if dcp_size > 1:
        # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
        attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
        attn_out_lse_all2all = torch.empty_like(attn_out_lse)
        dist.all_to_all_single(attn_out_lse_all2all, attn_out_lse, group=dcp_group)
        attn_out_lse = attn_out_lse_all2all.permute([2, 0, 1])

        # [DCP_DEBUG] Log shapes after all_to_all
        logger.info(
            "[DCP_DEBUG] _process_attn_out_lse AFTER all_to_all: "
            "attn_out_lse.shape=%s, "
            "attn_out_lse stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
            str(attn_out_lse.shape),
            attn_out_lse.mean().item(),
            attn_out_lse.std().item(),
            attn_out_lse.min().item(),
            attn_out_lse.max().item(),
        )

    if pcp_size > 1:
        # AllGather out&lse within CP group
        attn_out_lse = get_pcp_group().all_gather(attn_out_lse.contiguous(), dim=0)

    return attn_out_lse


def _npu_attention_update(head_size, attn_out_lse: torch.Tensor) -> torch.Tensor:
    pcp_size = get_pcp_group().world_size
    dcp_size = get_decode_context_model_parallel_world_size()
    # [PCP * S, DCP * H, D+1]
    B_total, H_total, D_plus_1 = attn_out_lse.shape
    S = B_total // pcp_size
    H = H_total // dcp_size
    D = head_size
    assert D_plus_1 == D + 1

    # [DCP_DEBUG] Log reshape parameters
    logger.info(
        "[DCP_DEBUG] _npu_attention_update: "
        "pcp_size=%d, dcp_size=%d, "
        "attn_out_lse.shape=%s, "
        "B_total=%d, H_total=%d, S=%d, H=%d, D=%d, "
        "view shape=[%d, %d, %d, %d, %d]",
        pcp_size,
        dcp_size,
        str(attn_out_lse.shape),
        B_total,
        H_total,
        S,
        H,
        D,
        pcp_size,
        S,
        dcp_size,
        H,
        D_plus_1,
    )

    # [PCP, S, DCP, H, D+1]
    x = attn_out_lse.view(pcp_size, S, dcp_size, H, D_plus_1)
    # [PCP, DCP, S, H, D+1]
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # Flatten [N, S, H, D+1], N = pcp_size * dcp_size
    x = x.view(-1, S, H, D_plus_1)
    # Split out lse
    out_flat, lse_flat = torch.split(x, [D, 1], dim=-1)  # [N, S, H, D], [N, S, H, 1]
    #    out: [N, S, H, D] -> [N, S*H, D]
    #    lse: [N, S, H, 1] -> [N, S*H]
    out_flat = out_flat.flatten(1, 2)  # [N, S*H, D]
    lse_flat = lse_flat.flatten(1, -1)  # [N, S*H]

    # [DCP_DEBUG] Log each partial result before npu_attention_update
    N = pcp_size * dcp_size
    for i in range(N):
        out_i = out_flat[i]
        lse_i = lse_flat[i]
        logger.info(
            "[DCP_DEBUG] _npu_attention_update partial[%d/%d]: "
            "out.shape=%s, out stats: mean=%.6f, std=%.6f, "
            "lse.shape=%s, lse stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
            i,
            N,
            str(out_i.shape),
            out_i.mean().item(),
            out_i.std().item(),
            str(lse_i.shape),
            lse_i.mean().item(),
            lse_i.std().item(),
            lse_i.min().item(),
            lse_i.max().item(),
        )

    #  unbind to list
    out_list = out_flat.unbind(0)  # [S*H, D]
    lse_list = lse_flat.unbind(0)  # [S*H]
    attn_out, _ = torch_npu.npu_attention_update(lse_list, out_list, 0)
    attn_out = attn_out.view(-1, H, D)

    # [DCP_DEBUG] Log final output
    logger.info(
        "[DCP_DEBUG] _npu_attention_update FINAL: attn_out.shape=%s, stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        str(attn_out.shape),
        attn_out.mean().item(),
        attn_out.std().item(),
        attn_out.min().item(),
        attn_out.max().item(),
    )

    return attn_out


def _npu_attn_out_lse_update(attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask):
    T = attn_out_mask.shape[0]
    N = attn_out_mask.shape[1]
    D = attn_out_mask.shape[2]
    attn_out_mask, attn_lse_mask = _out_lse_reshape(attn_out_mask, attn_lse_mask)
    attn_out_nomask, attn_lse_nomask = _out_lse_reshape(attn_out_nomask, attn_lse_nomask)
    attn_out_mask = attn_out_mask.to(torch.float32)
    attn_out_nomask = attn_out_nomask.to(torch.float32)
    attn_lse_mask = attn_lse_mask.to(torch.float32)
    attn_lse_nomask = attn_lse_nomask.to(torch.float32)
    attn_output = [attn_out_nomask, attn_out_mask]
    attn_lse = [attn_lse_nomask, attn_lse_mask]
    update_type = 0
    output, _ = torch_npu.npu_attention_update(attn_lse, attn_output, update_type)
    output = output.view(T, N, D)
    return output


def _out_lse_reshape(attn_out: torch.Tensor, attn_lse: torch.Tensor) -> torch.Tensor:
    attn_out = attn_out.contiguous().view(attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
    attn_lse = attn_lse.contiguous().view(attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
    return attn_out, attn_lse


def _update_out_and_lse(out_list: torch.Tensor, lse_list: torch.Tensor) -> torch.Tensor:
    """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
    Args:
        out_list: shape = [N, batch_size, num_heads, head_size]
        lse_list: shape = [N, batch_size, num_heads, 1]
    Returns:
        out_final: shape = [batch_size, num_heads, head_size]
        lse_final: shape = [batch_size, num_heads, 1]
    """
    lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
    out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list, dim=0)
    return out_final, lse_final
