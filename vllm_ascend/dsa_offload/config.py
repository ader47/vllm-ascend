# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 稀疏卸载的类型化拉起配置。

用户仍通过 ``additional_config["dsa_sparse_config"]`` 配置特性，但解析后
只保存在 ``AscendConfig.dsa_offload_config``。本模块不会向 vLLM 的
``CacheConfig`` 动态追加属性，也不参与请求或 layer 热路径。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from vllm_ascend.dsa_offload.contracts import (
    DSA_INDEX_HEAD_DIM,
    DSA_KV_LORA_RANK,
    DSA_LIDU_OUTPUT_CAPACITY,
    DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS,
    DSA_MTP_MAX_SPECULATIVE_TOKENS,
    DSA_MTP_SUPPORTED_RESIDENT_BUDGETS,
    DSA_NOMTP_SUPPORTED_RESIDENT_BUDGETS,
    DSA_QK_ROPE_HEAD_DIM,
    DSA_REQUIRED_CACHE_BLOCK_SIZE,
    DSA_SFA_COMPUTE_TOPK,
    DSA_SUPPORTED_INDEX_HEAD_COUNTS,
)
from vllm_ascend.dsa_offload.model_support import (
    DSAOffloadModelCapabilities,
    require_dsa_offload_model_support,
)

DSA_SPARSE_CONFIG_KEY = "dsa_sparse_config"
DSA_TRACE_POINT_FIRST_SAMPLE = "first_sample"
DSA_TRACE_POINTS = frozenset({DSA_TRACE_POINT_FIRST_SAMPLE})

_PUBLIC_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "split_indexer_cache",
        "indexer_mla_block_ratio",
        "sparse_activation_tokens",
        "prompt_budget_thresholds",
        "resident_budget_tokens",
        "max_active_reqs",
        "hot_cpu_block_multiple",
        "enable_row_mode_decode_graph",
        "trace_points",
    }
)
_TRACE_CONFIG_KEYS = frozenset({"enabled", "points", "ranks"})


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must be bool, got {type(value).__name__}")
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must be positive, got {value}")
    return value


def _positive_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must be numeric, got {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must be positive and finite, got {parsed}")
    return parsed


def _positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must be a sequence "
            f"of positive integers, got {type(value).__name__}"
        )
    parsed = tuple(_positive_int(item, field_name=field_name) for item in value)
    if not parsed:
        raise ValueError(f"{DSA_SPARSE_CONFIG_KEY}[{field_name!r}] must not be empty")
    return parsed


def _parse_csv_or_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _is_all_selector(value: Any) -> bool:
    return isinstance(value, str) and value in ("*", "all")


@dataclass(frozen=True)
class DSAOffloadTraceConfig:
    """拉起期解析完成、运行时只读的 DSA 调测配置。"""

    enabled: bool = False
    points: frozenset[str] = frozenset()
    ranks: frozenset[int] | None = None

    @classmethod
    def from_value(cls, raw_value: Any) -> DSAOffloadTraceConfig:
        if raw_value is None:
            return cls()
        if isinstance(raw_value, bool):
            raw_config: Mapping[str, Any] = {"enabled": raw_value}
        elif isinstance(raw_value, Mapping):
            raw_config = raw_value
        else:
            raise TypeError(
                f"{DSA_SPARSE_CONFIG_KEY}['trace_points'] must be bool or mapping, got {type(raw_value).__name__}"
            )

        unknown = sorted(set(raw_config) - _TRACE_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unknown DSA trace config keys: {unknown}; supported={sorted(_TRACE_CONFIG_KEYS)}")

        enabled = _strict_bool(
            raw_config.get("enabled", False),
            field_name="trace_points.enabled",
        )
        if not enabled:
            return cls()

        raw_points = raw_config.get("points")
        if raw_points is None or _is_all_selector(raw_points):
            points = DSA_TRACE_POINTS
        else:
            points = frozenset(str(item).strip() for item in _parse_csv_or_iterable(raw_points) if str(item).strip())
        unknown_points = sorted(points - DSA_TRACE_POINTS)
        if unknown_points:
            raise ValueError(f"Unknown DSA trace points: {unknown_points}; supported={sorted(DSA_TRACE_POINTS)}")

        raw_ranks = raw_config.get("ranks")
        if raw_ranks is None or _is_all_selector(raw_ranks):
            ranks = None
        else:
            ranks = frozenset(int(item) for item in _parse_csv_or_iterable(raw_ranks))
            if any(rank < 0 for rank in ranks):
                raise ValueError("DSA trace ranks must contain non-negative integers")
        return cls(enabled=True, points=points, ranks=ranks)


@dataclass(frozen=True)
class DSAOffloadConfig:
    """DSA offload 的唯一类型化配置真源。"""

    enabled: bool = False
    indexer_mla_block_ratio: int = 3
    sparse_activation_tokens: int = 6144
    prompt_budget_thresholds: tuple[int, ...] = (32768, 65536)
    resident_budget_tokens: tuple[int, ...] = (6144, 10240, 12288)
    max_active_reqs: int = 256
    hot_cpu_block_multiple: float = 3.0
    enable_row_mode_decode_graph: bool = False
    trace_points: DSAOffloadTraceConfig = field(default_factory=DSAOffloadTraceConfig)
    model_capabilities: DSAOffloadModelCapabilities | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    mtp_num_speculative_tokens: int = field(
        default=0,
        compare=False,
        repr=False,
    )
    mtp_cache_layer_count: int = field(
        default=0,
        compare=False,
        repr=False,
    )

    @property
    def split_indexer_cache(self) -> bool:
        """DSA offload 的固有布局约束，不再作为可独立关闭的运行模式。"""
        return self.enabled

    @property
    def max_resident_budget_tokens(self) -> int:
        return max(self.resident_budget_tokens)

    @property
    def mtp_enabled(self) -> bool:
        return self.mtp_num_speculative_tokens > 0

    @property
    def uniform_decode_query_len(self) -> int:
        return 1 + self.mtp_num_speculative_tokens

    @property
    def sparse_tail_block_count(self) -> int:
        """返回 sparse resident 固定持有的尾块数。"""

        return 2 if self.mtp_enabled else 1

    def is_mtp_cache_layer_index(
        self,
        layer_index: int,
        target_layer_count: int,
    ) -> bool:
        """判断物理层下标是否属于全量驻留的 MTP Cache layer。"""

        layer_index = int(layer_index)
        target_layer_count = int(target_layer_count)
        return self.mtp_enabled and target_layer_count <= layer_index < target_layer_count + self.mtp_cache_layer_count

    @classmethod
    def from_dict(
        cls,
        raw_config: Any,
        *,
        vllm_config: Any | None = None,
    ) -> DSAOffloadConfig:
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, Mapping):
            raise TypeError(
                f"additional_config[{DSA_SPARSE_CONFIG_KEY!r}] must be a mapping, got {type(raw_config).__name__}"
            )

        unknown = sorted(set(raw_config) - _PUBLIC_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown {DSA_SPARSE_CONFIG_KEY} keys: {unknown}; supported={sorted(_PUBLIC_CONFIG_KEYS)}"
            )

        enabled = _strict_bool(
            raw_config.get("enabled", False),
            field_name="enabled",
        )
        if "split_indexer_cache" in raw_config:
            split_indexer_cache = _strict_bool(
                raw_config["split_indexer_cache"],
                field_name="split_indexer_cache",
            )
            if enabled and not split_indexer_cache:
                raise ValueError(
                    "DSA sparse offload always requires a split Indexer/MLA "
                    "cache layout; split_indexer_cache cannot be false when "
                    "enabled"
                )

        config = cls(
            enabled=enabled,
            indexer_mla_block_ratio=_positive_int(
                raw_config.get("indexer_mla_block_ratio", 3),
                field_name="indexer_mla_block_ratio",
            ),
            sparse_activation_tokens=_positive_int(
                raw_config.get("sparse_activation_tokens", 6144),
                field_name="sparse_activation_tokens",
            ),
            prompt_budget_thresholds=_positive_int_tuple(
                raw_config.get(
                    "prompt_budget_thresholds",
                    (32768, 65536),
                ),
                field_name="prompt_budget_thresholds",
            ),
            resident_budget_tokens=_positive_int_tuple(
                raw_config.get(
                    "resident_budget_tokens",
                    (6144, 10240, 12288),
                ),
                field_name="resident_budget_tokens",
            ),
            max_active_reqs=_positive_int(
                raw_config.get("max_active_reqs", 256),
                field_name="max_active_reqs",
            ),
            hot_cpu_block_multiple=_positive_float(
                raw_config.get("hot_cpu_block_multiple", 3.0),
                field_name="hot_cpu_block_multiple",
            ),
            enable_row_mode_decode_graph=_strict_bool(
                raw_config.get("enable_row_mode_decode_graph", False),
                field_name="enable_row_mode_decode_graph",
            ),
            trace_points=DSAOffloadTraceConfig.from_value(raw_config.get("trace_points")),
        )
        config._validate_static_contract()
        if config.enabled and vllm_config is not None:
            (
                capabilities,
                mtp_num_speculative_tokens,
                mtp_cache_layer_count,
            ) = config._validate_runtime_contract(vllm_config)
            config = replace(
                config,
                model_capabilities=capabilities,
                mtp_num_speculative_tokens=mtp_num_speculative_tokens,
                mtp_cache_layer_count=mtp_cache_layer_count,
            )
        return config

    def _validate_static_contract(self) -> None:
        thresholds = self.prompt_budget_thresholds
        budgets = self.resident_budget_tokens
        if len(budgets) != len(thresholds) + 1:
            raise ValueError("resident_budget_tokens must contain exactly one more entry than prompt_budget_thresholds")
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("prompt_budget_thresholds must be strictly increasing")
        if any(left > right for left, right in zip(budgets, budgets[1:])):
            raise ValueError("resident_budget_tokens must be non-decreasing")
        if budgets[0] < DSA_SFA_COMPUTE_TOPK:
            raise ValueError("The smallest resident budget must cover SFA-Offload topK")
        if budgets[-1] > DSA_LIDU_OUTPUT_CAPACITY:
            raise ValueError("The largest resident budget exceeds LIDU output capacity")
        unsupported = tuple(budget for budget in budgets if budget not in DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS)
        if unsupported:
            raise ValueError(
                "Resident budgets are not supported by the current LIDU "
                f"kernel: unsupported={unsupported}, supported="
                f"{DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS}"
            )
        if self.sparse_activation_tokens > budgets[0]:
            raise ValueError("sparse_activation_tokens cannot exceed the smallest resident budget")

    def _validate_runtime_contract(
        self,
        vllm_config: Any,
    ) -> tuple[DSAOffloadModelCapabilities, int, int]:
        capabilities = require_dsa_offload_model_support(vllm_config.model_config)
        operator_contract = {
            "index_head_dim": (
                capabilities.index_head_dim,
                DSA_INDEX_HEAD_DIM,
            ),
            "kv_lora_rank": (
                capabilities.kv_lora_rank,
                DSA_KV_LORA_RANK,
            ),
            "qk_rope_head_dim": (
                capabilities.qk_rope_head_dim,
                DSA_QK_ROPE_HEAD_DIM,
            ),
        }
        mismatched = {
            name: {"actual": actual, "required": required}
            for name, (actual, required) in operator_contract.items()
            if actual != required
        }
        if capabilities.index_num_heads not in DSA_SUPPORTED_INDEX_HEAD_COUNTS:
            mismatched["index_n_heads"] = {
                "actual": capabilities.index_num_heads,
                "required": DSA_SUPPORTED_INDEX_HEAD_COUNTS,
            }
        if mismatched:
            raise ValueError(
                f"DSA offload model dimensions do not match the current LIDU/KSC/SFA-Offload ABI: {mismatched}"
            )
        if capabilities.indexer_types is not None:
            hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
            num_hidden_layers = getattr(hf_text_config, "num_hidden_layers", None)
            if num_hidden_layers is None or len(capabilities.indexer_types) != num_hidden_layers:
                raise ValueError(
                    "DSA shared-indexer topology must declare one indexer type "
                    f"per hidden layer: indexer_types={len(capabilities.indexer_types)}, "
                    f"num_hidden_layers={num_hidden_layers!r}"
                )
        if capabilities.has_shared_indexer_layers:
            if not capabilities.full_indexer_layer_indices:
                raise ValueError(
                    "DSA shared-indexer topology requires at least one full indexer layer to source top-K selection"
                )
            first_shared = capabilities.shared_indexer_layer_indices[0]
            first_full = capabilities.full_indexer_layer_indices[0]
            if first_shared < first_full:
                raise ValueError(
                    "DSA shared-indexer topology requires every shared layer "
                    "to follow a full source layer: "
                    f"first_shared={first_shared}, first_full={first_full}"
                )
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        observability_config = vllm_config.observability_config

        max_num_seqs = int(scheduler_config.max_num_seqs or 0)
        if max_num_seqs > self.max_active_reqs:
            raise ValueError(
                "DSA request-row capacity must cover max_num_seqs: "
                f"max_active_reqs={self.max_active_reqs}, "
                f"max_num_seqs={max_num_seqs}"
            )

        if scheduler_config.async_scheduling is not False:
            raise ValueError("DSA sparse offload currently requires async_scheduling=False")
        chunked_prefill_enabled = (
            bool(scheduler_config.enable_chunked_prefill) or int(scheduler_config.long_prefill_token_threshold or 0) > 0
        )
        if chunked_prefill_enabled and not bool(scheduler_config.scheduler_reserve_full_isl):
            raise ValueError(
                "DSA chunked prefill requires scheduler_reserve_full_isl=True "
                "so the complete prompt can be admitted into every active "
                "full-context cache group before its first chunk"
            )
        if bool(cache_config.enable_prefix_caching):
            raise ValueError("DSA sparse offload does not yet support prefix caching")
        speculative_config = vllm_config.speculative_config
        data_parallel_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
        if data_parallel_size > 1:
            if not self.enable_row_mode_decode_graph and not bool(vllm_config.model_config.enforce_eager):
                raise ValueError(
                    "DSA sparse offload DP requires true eager execution or enable_row_mode_decode_graph=True"
                )
            if not bool(getattr(parallel_config, "enable_expert_parallel", False)):
                raise ValueError("DSA sparse offload DP requires enable_expert_parallel=True")
            if (
                getattr(parallel_config, "data_parallel_backend", "mp") != "mp"
                or getattr(
                    parallel_config,
                    "distributed_executor_backend",
                    None,
                )
                == "external_launcher"
            ):
                raise ValueError("DSA sparse offload DP requires internal multiprocessing data parallelism")
            if bool(getattr(parallel_config, "data_parallel_external_lb", False)) or bool(
                getattr(parallel_config, "data_parallel_hybrid_lb", False)
            ):
                raise ValueError("DSA sparse offload DP does not support external or hybrid load balancing")
        mtp_num_speculative_tokens = 0
        mtp_cache_layer_count = 0
        if speculative_config is not None:
            method = getattr(speculative_config, "method", None)
            mtp_num_speculative_tokens = int(getattr(speculative_config, "num_speculative_tokens", 0) or 0)
            if method != "mtp":
                raise ValueError(f"DSA sparse offload initially supports only speculative method='mtp', got {method!r}")
            if mtp_num_speculative_tokens != DSA_MTP_MAX_SPECULATIVE_TOKENS:
                raise ValueError(
                    "DSA compromise MTP requires exactly "
                    f"num_speculative_tokens={DSA_MTP_MAX_SPECULATIVE_TOKENS}, "
                    f"got {mtp_num_speculative_tokens}"
                )
            if bool(getattr(speculative_config, "parallel_drafting", False)):
                raise ValueError("DSA compromise MTP does not support parallel_drafting")
            if bool(
                getattr(
                    speculative_config,
                    "disable_padded_drafter_batch",
                    False,
                )
            ):
                raise ValueError("DSA compromise MTP requires padded drafter batches")
            draft_model_config = speculative_config.draft_model_config
            draft_hf_config = draft_model_config.hf_config
            mtp_cache_layer_count = int(getattr(draft_hf_config, "num_nextn_predict_layers", 0) or 0)
            if mtp_cache_layer_count <= 0:
                raise ValueError(
                    "DSA compromise MTP requires at least one physical "
                    "MTP cache layer, got "
                    f"num_nextn_predict_layers={mtp_cache_layer_count}"
                )
            unsupported_mtp_budgets = tuple(
                budget for budget in self.resident_budget_tokens if budget not in DSA_MTP_SUPPORTED_RESIDENT_BUDGETS
            )
            if unsupported_mtp_budgets:
                raise ValueError(
                    "DSA compromise MTP resident budgets must be one of "
                    f"{DSA_MTP_SUPPORTED_RESIDENT_BUDGETS}; unsupported="
                    f"{unsupported_mtp_budgets}"
                )
            mtp_min_activation_tokens = DSA_MTP_SUPPORTED_RESIDENT_BUDGETS[0]
            if self.sparse_activation_tokens < mtp_min_activation_tokens:
                raise ValueError(
                    "DSA compromise MTP sparse_activation_tokens must be at "
                    "least the minimum MTP resident tier: "
                    "required="
                    f"{mtp_min_activation_tokens}, "
                    f"got={self.sparse_activation_tokens}"
                )
        else:
            unsupported_nomtp_budgets = tuple(
                budget for budget in self.resident_budget_tokens if budget not in DSA_NOMTP_SUPPORTED_RESIDENT_BUDGETS
            )
            if unsupported_nomtp_budgets:
                raise ValueError(
                    "DSA non-MTP resident budgets must be one of "
                    f"{DSA_NOMTP_SUPPORTED_RESIDENT_BUDGETS}; unsupported="
                    f"{unsupported_nomtp_budgets}"
                )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError("DSA sparse offload does not yet support KV transfer connectors")
        if parallel_config.decode_context_parallel_size != 1 or parallel_config.prefill_context_parallel_size != 1:
            raise ValueError("DSA split Indexer/MLA cache does not yet support decode or prefill context parallelism")
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError("DSA split Indexer/MLA cache does not yet support pipeline parallelism")
        if observability_config.kv_cache_metrics:
            raise ValueError(
                "DSA split Indexer/MLA cache does not yet support KV-cache "
                "metrics because independent block pools reuse block IDs"
            )
        kv_events_config = vllm_config.kv_events_config
        if kv_events_config is not None and kv_events_config.enable_kv_cache_events:
            raise ValueError(
                "DSA split Indexer/MLA cache does not yet support KV-cache "
                "events because independent block pools reuse block IDs"
            )
        if self.enable_row_mode_decode_graph and bool(vllm_config.model_config.enforce_eager):
            raise ValueError("enable_row_mode_decode_graph requires enforce_eager=False")
        return (
            capabilities,
            mtp_num_speculative_tokens,
            mtp_cache_layer_count,
        )

    def validate_finalized_cache_contract(self, vllm_config: Any) -> None:
        """校验经过 Ascend 后端刷新后的物理 cache 块契约。

        v0.23 的 ``NPUPlatform.check_and_update_config`` 会先构造
        ``AscendConfig``，随后才执行 ``refresh_block_size``。因此块大小及
        block 对齐约束必须在后者完成后检查，不能在配置解析阶段依据临时值
        作出错误结论。
        """
        if not self.enabled:
            return

        block_size = int(vllm_config.cache_config.block_size or 0)
        if block_size != DSA_REQUIRED_CACHE_BLOCK_SIZE:
            raise ValueError(
                "The initial DSA offload migration requires block_size="
                f"{DSA_REQUIRED_CACHE_BLOCK_SIZE}, got {block_size}"
            )
        if any(
            value % block_size != 0
            for value in (
                self.sparse_activation_tokens,
                *self.resident_budget_tokens,
            )
        ):
            raise ValueError(
                f"sparse_activation_tokens and resident_budget_tokens must be aligned to block_size={block_size}"
            )

    def validate_finalized_graph_contract(
        self,
        vllm_config: Any,
        *,
        phase: str,
        require_resolved_mode: bool = False,
    ) -> None:
        """在基线完成图模式归一化后校验 DSA 的最终执行合同。

        ``VllmConfig`` 和 Ascend attention backend 都可能继续调整用户输入的
        compilation/cudagraph 配置，所以不能只在解析
        ``dsa_sparse_config`` 时依据原始参数作判断。平台归一化和 backend
        capability 决议后各调用一次本函数，确保错误配置在 capture 前失败。
        """

        if not self.enabled:
            return

        from vllm.config import CompilationMode, CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        cudagraph_mode = compilation_config.cudagraph_mode
        if self.enable_row_mode_decode_graph:
            if compilation_config.mode != CompilationMode.VLLM_COMPILE:
                raise ValueError(
                    "DSA row-mode decode graph requires "
                    "compilation_config.mode=VLLM_COMPILE after Ascend "
                    f"normalization: phase={phase}, "
                    f"mode={compilation_config.mode}"
                )
            if not cudagraph_mode.has_full_cudagraphs():
                raise ValueError(
                    "DSA row-mode decode graph requires a cudagraph mode "
                    "with FULL decode graphs after Ascend normalization: "
                    f"phase={phase}, cudagraph_mode={cudagraph_mode}"
                )
            if require_resolved_mode and cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                raise ValueError(
                    "DSA row-mode decode graph requires "
                    "cudagraph_mode=FULL_DECODE_ONLY after attention-backend "
                    "resolution: "
                    f"phase={phase}, cudagraph_mode={cudagraph_mode}"
                )
            capture_sizes = compilation_config.cudagraph_capture_sizes
            if not capture_sizes:
                raise ValueError(
                    "DSA row-mode decode graph requires non-empty "
                    "cudagraph_capture_sizes after Ascend normalization: "
                    f"phase={phase}"
                )
            return

        if cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(
                "DSA sparse offload with "
                "enable_row_mode_decode_graph=False requires true eager "
                "execution (cudagraph_mode=NONE). Set enforce_eager=True or "
                "enable the DSA row-mode decode graph: "
                f"phase={phase}, cudagraph_mode={cudagraph_mode}"
            )
