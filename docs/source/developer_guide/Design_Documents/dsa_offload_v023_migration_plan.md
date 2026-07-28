# DSA 稀疏卸载 v0.23 迁移计划

> - 最后更新：2026-07-28
> - 当前阶段：P0/P1 已完成本地实现与单测，等待 910C 拉起验证
> - 迁移目标：只修改 vLLM-Ascend，不修改 vLLM

## 1. 文档职责

本文档是 DSA 稀疏卸载从 vLLM-Ascend v0.19.1 原型迁移到 v0.23.0
基线的持续维护记录，负责固定以下信息：

- 已确认的功能语义和首版支持边界；
- 新旧基线的关键差异；
- 各迁移阶段的优先级、设计约束、验收门槛和当前状态；
- v0.19 临时 patch 机制到 v0.23 原生扩展点的映射；
- 需要在 910C、后续 A5 环境完成的回归项目；
- 尚未决策的问题及其最晚决策阶段。

每完成一个阶段，都必须同步更新本文档的状态表、验收结果和变更记录。
v0.19 代码只作为语义参考，不能机械搬运其 patch 拓扑。

## 2. 基线与目录

| 角色 | 本地仓库 | 基线提交 |
|---|---|---|
| v0.19 vLLM 参考 | `latest_vllm_for_dsa_migration/vllm` | `b1388b1f` |
| v0.19 DSA 实现参考 | `latest_vllm_for_dsa_migration/vllm-ascend` | 基于 `da421afa` 的 custom 分支 |
| v0.23 vLLM 参考 | `DSA_Proj/vllm-v0.23` | `0fc695fc` |
| v0.23 迁移目标 | `DSA_Proj/vllm-ascend-v0.23` | `d19edec6` |

## 3. 总体迁移原则

1. **vLLM 零修改**：所有实现进入 vLLM-Ascend。
2. **迁移语义，不迁移旧 patch 形态**：优先使用 v0.23 已有扩展点。
3. **不复制 Scheduler 主循环**：不得复制或整体替换 `schedule()`。
4. **eager/graph 共用语义真源**：eager 使用 active-prefix view，graph
   使用 captured-prefix + PAD view，底层状态由同一 owner 管理。
5. **非 DSA 路径保持基线行为**：DSA 未开启时，不改变 cache 布局、容量、
   调度和执行路径。
6. **启动期显式失败**：用户开启 DSA 后，遇到不支持配置必须报错，不允许
   静默退化成“未卸载但看似跑通”的路径。
7. **框架 ABI 与设备实现解耦**：先在 910C 验证框架语义；A5 算子迁移不能
   迫使框架侧改变同一套接口。
8. **热路径不引入临时元数据搬运**：禁止新增逐 step 的 list-to-tensor、
   `.item()`、D2H/H2D 往返和按请求拆子 batch。

## 4. 已冻结的 DSA 功能语义

### 4.1 Cache 物理布局

- Indexer cache 保存完整上下文并驻留 HBM。
- MLA resident cache 只保存当前请求的 resident budget 和尾块。
- 满块 MLA cache 通过固定 DRAM arena 管理；arena 拉起后不扩容，地址稳定。
- Indexer/MLA 的 block 分配、slot mapping 和容量必须分别计算，不能继续把
  Indexer 字节隐含在 MLA page size 中。

### 4.2 请求阶段

首版保持四种逐行状态：

| 状态 | 含义 |
|---|---|
| `DENSE` | 尚未触发稀疏卸载，SFA 能看到当前完整有效上下文 |
| `ENTER` | 本轮完成 dense 到 sparse 的资源和元数据转换 |
| `SPARSE` | 使用固定 resident budget、尾块和 DRAM 候选执行稀疏 decode |
| `PAD` | 图捕获补齐行；所有算子必须安全空转 |

一批请求可以包含任意顺序的不同状态，不允许在 Python 侧按状态拆成多个
子 batch。

### 4.3 Budget 语义

- 请求在 admission 时根据 **prompt token 数** 冻结目标 resident budget。
- decode 期间不随上下文增长换档。
- 当前公开默认配置：

  - `sparse_activation_tokens=6144`
  - `prompt_budget_thresholds=(32768, 65536)`
  - `resident_budget_tokens=(6144, 10240, 12288)`

- 当前 LIDU 算子 ABI 只支持上述三档 resident budget，最大输出容量为
  `12288`。
- SFA-Offload 实际计算 topK 固定为 `2048`；更大的 resident budget 用于
  降低跨 step miss 和 DRAM 到 HBM 的换入量。

### 4.4 算子流水

稳定 sparse decode 的目标流水为：

1. LIDU：计算重要 token，并更新 tokenwise resident 映射；
2. KSC：按照 LIDU 给出的 miss prefix 完成 DRAM 到 HBM 换入；
3. SFA-Offload：消费 topK 与尾块完成注意力计算；
4. Full-block dump：在本轮产生新满块时，按固定元数据将 HBM MLA 块写入
   DRAM arena。

所有算子在 graph captured rows 中都必须支持 `PAD` 或空转哨兵。

## 5. v0.23 基线差异结论

### 5.1 原生 Indexer/MLA 布局

v0.23 的 Ascend 实现有意忽略原始 `DeepseekV32IndexerCache` spec：

- `IndexerWrapper` 清空原始 `k_cache`；
- `NPUModelRunner.get_kv_cache_spec()` 跳过
  `DeepseekV32IndexerCache`；
- Indexer K 被编码进 `AscendMLAAttentionSpec.sparse_head_dim`；
- MLA latent、RoPE、Indexer K 和可选量化 scale 共用一个物理 cache group。

这是 v0.23 原生 SFA 的打包设计，不是缺陷。DSA 要改变这一物理不变量，
必须原子地完成 spec、字节核算、tensor 分配、block table、cache 绑定和
算子消费的解耦，不能只恢复一个 Indexer spec。

### 5.2 KV cache 管控面

- `KVCacheConfig.num_blocks` 仍是标量；
- `KVCacheTensor.size` 可以表达不同大小的物理 tensor；
- `KVCacheSpecRegistry` 支持平台注册自有 spec 与 manager；
- vLLM-Ascend 已有 coordinator factory、KV 分组和配置计算扩展点；
- vLLM-Ascend 已有 `NPUInputBatch` 和 `MultiGroupBlockTable`。

P2 的当前方向是：将 `num_blocks` 保持为 MLA 基础容量，以配置 ratio
计算 Indexer tensor 容量；各物理池的最终 block 数由 finalized
tensor/spec 尺寸推导，避免重新引入 v0.19 的动态 `CacheConfig` 属性。

### 5.3 配置初始化时序

v0.23 的平台流程先构造 `AscendConfig`，后执行 `refresh_block_size()`。
因此配置校验分为两段：

1. `AscendConfig` 构造时校验模型能力、调度模式和功能组合；
2. `refresh_block_size()` 后校验最终 `block_size` 和 budget 对齐。

不得在第一阶段使用尚未刷新的临时 block size 作出结论。

### 5.4 Scheduler 边界

按以下顺序选择方案：

1. 如果 coordinator/manager 与共享 planner 足以派生状态，不新增 DSA
   Scheduler；
2. 如果 request target budget 或生命周期确实无法从现有接口派生，只增加
   很薄的 Ascend 生命周期适配，并始终调用 `super()`；
3. 禁止复制 `schedule()`，禁止全局替换
   `SchedulerOutput/NewRequestData/CachedRequestData`。

### 5.5 Worker 与图模式边界

- 跨 step 的行状态归 `NPUInputBatch`；
- 多物理 cache 的块表行为归 `MultiGroupBlockTable`；
- 输入和 slot 元数据扩展 v0.23 现有 `NPUModelRunner` 流程；
- attention view 扩展 `AscendSFAMetadataBuilder`；
- DSA 图准入只缩小原生 FULL decode 的准入范围，不另造图运行时。

## 6. 首版支持矩阵

首个可运行版本明确要求：

| 维度 | 首版要求 |
|---|---|
| 主验收模型 | GLM-5.1 |
| 强制回归模型 | DeepSeek-V3.2 |
| 设备 | Ascend 910C |
| `block_size` | `128` |
| `async_scheduling` | `False` |
| prefix caching | 关闭 |
| chunked prefill | 关闭 |
| speculative/MTP | 关闭 |
| KV transfer connector | 关闭 |
| 图模式 | P6 完成前仅验证 eager |

这些限制用于隔离核心迁移，不代表最终能力边界。每解除一项限制，必须追加
独立测试。

## 7. P0 回归合同

### 7.1 功能用例

| ID | 场景 | 关键观察 |
|---|---|---|
| F01 | bsz=1，超短 prompt，不足一个满块 | 保持 DENSE，不越界 |
| F02 | bsz=4，全部短序列 | 不触发稀疏 IO，输出与基线一致 |
| F03 | bsz=4，长短序列混合 | 不拆子 batch；各行阶段独立 |
| F04 | prompt 覆盖三档阈值 | admission 后 budget 正确且 decode 不换档 |
| F05 | 短 prompt 长 decode | 新满块持续 dump，达到阈值后进入 sparse |
| F06 | 长 prompt 首次 decode | prefill 满块已可见，首轮 sparse 结果正确 |
| F07 | decode 跨越新满块边界 | dump 当轮正确，后续 LIDU/KSC 可消费 |
| F08 | batch 内 decode 轮次不同 | 行状态与 slot mapping 不串行污染 |
| F09 | continuous batching 行复用 | reset/append 后旧请求元数据不可见 |
| F10 | 大 batch 请求结束 | DRAM 引用释放正确，无百毫秒级逐块释放 |

### 7.2 图模式用例

| ID | 场景 | 关键观察 |
|---|---|---|
| G01 | 全 DENSE decode | 使用统一 row-mode 图 |
| G02 | 全 SPARSE decode | LIDU/KSC/SFA-Offload 正确 replay |
| G03 | DENSE/ENTER/SPARSE/PAD 混合 | 单图、任意行顺序、无子 batch |
| G04 | decode 新满块 dump | 固定地址元数据 replay 正确 |
| G05 | captured batch 大于 active batch | PAD 行所有算子安全空转 |
| G06 | 请求完成后 captured row 复用 | 图 buffer 不携带旧状态 |

### 7.3 每次服务器验证保留的工件

- 完整拉起配置快照；
- prompt token 数、请求 budget 和阶段转换记录；
- 固定抽样请求的输出 token IDs/finish reason；
- Indexer/MLA/DRAM 容量报告；
- eager 与 graph 的结果 JSONL；
- 性能回归时保留 profiler 数据目录及场景参数。

P0 当前状态：合同与矩阵已冻结；910C golden 工件待在 P2 数据面接通前后各
保存一份。

## 8. 阶段状态

| 阶段 | 优先级 | 内容 | 当前状态 |
|---|---:|---|---|
| P0 | 0 | 语义、支持矩阵、回归合同、迁移记录 | 合同完成，golden 待采集 |
| P1 | 0 | 类型化配置与能力型模型识别 | 本地实现完成，待 910C 拉起 |
| P2 | 0 | Indexer/MLA spec、字节规划、物理 tensor 解耦 | 未开始 |
| P3 | 0 | 独立 manager/coordinator 与生命周期合同 | 未开始 |
| P4 | 1 | `NPUInputBatch`、block table、统一行状态 | 未开始 |
| P5 | 1 | eager 数据面：dump、LIDU、KSC、SFA-Offload | 未开始 |
| P6 | 1 | 复用原生图捕获/replay | 未开始 |
| P7 | 2 | 清理、场景扩展、A5 算子验证 | 未开始 |

## 9. P1：类型化配置与模型能力

### 9.1 已实现内容

- 用户入口保持 `additional_config["dsa_sparse_config"]`；
- 解析结果为冻结的 `DSAOffloadConfig`；
- 唯一持有者为 `AscendConfig.dsa_offload_config`；
- 不向 vLLM `CacheConfig` 动态挂载 DSA 属性；
- 未知字段、错误类型和冲突配置在拉起期失败；
- `hot_cpu_block_multiple` 保留浮点语义，P2 计算块数时再向上取整；
- 模型支持依据 MLA、sparse-indexer、topK 和维度能力判断，不使用架构名
  白名单；
- GLM-5.1、DeepSeek-V3.2 和未来满足能力协议的模型共用一条路径；
- block size 校验遵守 v0.23 平台刷新时序。

### 9.2 首版配置示例

```python
additional_config = {
    "dsa_sparse_config": {
        "enabled": True,
        "split_indexer_cache": True,
        "indexer_mla_block_ratio": 3,
        "sparse_activation_tokens": 6144,
        "prompt_budget_thresholds": [32768, 65536],
        "resident_budget_tokens": [6144, 10240, 12288],
        "max_active_reqs": 256,
        "hot_cpu_block_multiple": 3.0,
        "enable_row_mode_decode_graph": False,
        "trace_points": {
            "enabled": False,
            "points": ["first_sample"],
            "ranks": [0],
        },
    },
}
```

`split_indexer_cache` 在首版中是 DSA 的固有约束。该字段暂时保留在公开配置
中用于明确表达布局；当 `enabled=True` 时，将其配置为 `False` 会在启动期
报错。

### 9.3 P1 验收

- 本地 Ruff：通过；
- 配置与模型能力单测：24 项通过；
- `AscendConfig` 独立导入：通过，无循环导入；
- 服务器 smoke 入口：`examples/dsa_demo/simple_prompt_test.py`；
- 910C 待验证：

  1. DSA 关闭时可正常拉起原生模型；
  2. DSA 开启且不兼容配置时给出明确错误；
  3. GLM-5.1 与 DeepSeek-V3.2 能正确识别能力；
  4. `block_size` 在 Ascend 刷新后再校验。

## 10. P2：物理 cache 解耦计划

P2 只处理“空间是什么、占多少、如何绑定”，不提前迁移请求阶段和算子热
路径。

### 10.1 设计步骤

1. 在 vLLM-Ascend 注册 DSA 专用 Indexer spec 和 manager 类型。
2. 在 DSA 模式下，`NPUModelRunner.get_kv_cache_spec()` 显式产生：

   - 全量 Indexer dense plane；
   - MLA resident plane；
   - 非 DSA 模式继续产生原生 packed `AscendMLAAttentionSpec`。

3. DSA resident MLA spec 的 page size 只计算 MLA latent、RoPE 和必要
   scale，不再包含 Indexer K。
4. Indexer tensor 和 MLA tensor 独立计算字节数与 block 容量。
5. 只从 finalized `KVCacheTensor.size`/spec 推导物理容量；不在
   `CacheConfig` 增加 `dsa_num_blocks` 一类影子字段。
6. 输出确定性的容量报告，并在所有 rank 核对相同的 group 顺序和容量。
7. 设计并测试 cache binding：每层 attention/indexer 必须绑定到正确 tensor，
   禁止沿用 packed tuple 的位置假设。

### 10.2 P2 禁止事项

- 只恢复 `DeepseekV32IndexerCache.get_kv_cache_spec()` 而不修改 consumer；
- 在 `AscendMLAAttentionSpec.page_size_bytes` 中同时计算两套 Indexer 空间；
- 用同一个标量 block 数假装两个物理池容量相同；
- 为了快速跑通而复制 v0.19 `patch_kv_cache_utils.py`。

### 10.3 P2 验收门槛

- DSA 关闭时原生 packed layout 完全不变；
- DSA 开启时不存在 Indexer 双重分配；
- tensor shape、page bytes、总分配字节和报告四者一致；
- TP 各 rank 的 group 顺序与容量一致；
- GLM-5.1 和 DeepSeek-V3.2 均通过 cache 初始化；
- 仅完成空间初始化尚不能宣称 DSA 推理可用。

## 11. P3：分配与生命周期计划

### 11.1 首选结构

- 使用 vLLM-Ascend coordinator factory 创建 DSA coordinator；
- Indexer manager 维护完整上下文 block；
- MLA resident manager 维护 sparse budget、保留尾块和阶段转换所需空间；
- admission 进行 component-wise 容量检查，不能只看某个合并比例；
- target resident budget 在请求 admission 时冻结。

### 11.2 Scheduler 决策门

P3 实现前逐项回答：

| 问题 | 可以由 coordinator/现有数据派生时 | 无法派生时 |
|---|---|---|
| target budget | planner 内计算并持久化 | 增加最小 request 生命周期字段 |
| 两组 block 分配 | coordinator/manager 完成 | 增加薄调度适配 |
| 新满块识别 | 由 logical length 与 block table 派生 | 输出最小 tensor/list 元数据 |
| preempt/resume | 首版直接拒绝 | 后续单独设计 DRAM ledger |

只有右列确实出现时才引入 Scheduler 适配，而且必须复用基线 `schedule()`。

### 11.3 P3 验收门槛

- add、decode grow、ENTER、free 的两组 block 账本一致；
- 首版不支持 preemption 时必须在启动期或 admission 明确拒绝；
- 不新增整套 SchedulerOutput 类型；
- 不复制调度循环；
- 不在 worker 各 rank 独立决定逻辑 block 分配。

## 12. P4-P7 摘要

### P4：统一行状态

将跨 step 状态纳入 `NPUInputBatch` 的类型化固定容量列；eager 和 graph 使用
同一 owner 的不同 view。`MultiGroupBlockTable` 承担多物理池块表。

### P5：eager 数据面

按顺序接通 prefill dump、DENSE、ENTER、SPARSE、decode dump，以及
LIDU/KSC/SFA-Offload。每一步先做正确性，再做 host 热路径优化。

### P6：图模式

复用 v0.23 原生 FULL decode capture/replay。DSA 只增加固定 buffer、逐行
状态和准入条件；不得创建第二套 graph dispatcher 或 graph-only 语义字段。

### P7：扩展

依次评估 chunked prefill、prefill/decode mixed、preemption、prefix cache、
MTP、async scheduling、KV transfer 和 A5 算子实现。

## 13. v0.19 机制迁移映射

| v0.19 机制 | v0.23 目标 |
|---|---|
| 动态 `CacheConfig` 属性 | `AscendConfig.dsa_offload_config` |
| 架构名白名单 | 模型能力协议 |
| 全局 Scheduler monkey patch | coordinator/manager，必要时薄生命周期适配 |
| 全局替换 SchedulerOutput 类 | 现有输出结构或最小 Ascend 扩展 |
| patch `SingleTypeKVCacheManager` | `KVCacheSpecRegistry` 注册 manager |
| 独立 DSA graph buffers/dispatcher | `NPUInputBatch` + 原生 graph buffer/view |
| 复制 `_update_states/_prepare_inputs` | 在 vLLM-Ascend 原生方法内增加局部 DSA 行为 |
| packed MLA+Indexer page size | 独立 Indexer/MLA spec 与 tensor |

## 14. 明确不迁移的旧实现

- v0.19 全局 Scheduler monkey patch；
- Scheduler 输出类型重绑定；
- 全局 `SingleTypeKVCacheManager` patch；
- `dsa_num_blocks` 等动态 cache 属性；
- 复制版 `_update_states` 或 `_prepare_inputs`；
- 独立 DSA graph-buffer/dispatcher 层级；
- 仅按模型 architecture 名称使能；
- 已退役 GS 数据面及其兼容 fallback。

## 15. 未决问题

| 问题 | 最晚决策阶段 | 当前方向 |
|---|---|---|
| group 容量编码在 spec 还是由 tensor size 推导 | P2 | 优先由 finalized tensor size 推导 |
| 是否需要薄 Scheduler 生命周期适配 | P3 | 默认不需要，按证据决定 |
| DSA 状态是否进入 SchedulerOutput | P3/P4 | 优先由共享 planner 和既有 block IDs 派生 |
| preempt 后 tokenwise row 与 DRAM ledger 生命周期 | P7 | row 释放，DRAM ledger 独立保留 |
| prefix/content hash 的 DRAM 身份 | P7 | 首版使用 request lifetime + logical block |
| A5 算子实现 | P7 | 框架 ABI 保持不变，由算子侧适配 |

## 16. 变更记录

- 2026-07-28：

  - 创建 v0.23 迁移计划；
  - 冻结首版支持矩阵和 P0 回归合同；
  - 完成 P1 类型化配置、模型能力识别和两阶段 block-size 校验；
  - 增加 P1 disabled/enabled/reject-async 三模式服务器 smoke demo；
  - 明确 P2/P3 的实现边界与验收门槛。
