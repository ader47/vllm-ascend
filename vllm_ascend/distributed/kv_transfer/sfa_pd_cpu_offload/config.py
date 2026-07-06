from __future__ import annotations

from vllm.config import VllmConfig

_DEFAULT_OFFLOAD_TP_RANK = 0


def get_offload_tp_rank(vllm_config: VllmConfig) -> int:
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    offload_tp_rank = int(extra_config.get("offload_tp_rank", _DEFAULT_OFFLOAD_TP_RANK))
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    if offload_tp_rank < 0 or offload_tp_rank >= tp_size:
        raise ValueError(
            "SFAPDCpuOffloadConnector offload_tp_rank must be in "
            f"[0, {tp_size}), got {offload_tp_rank}"
        )
    return offload_tp_rank
