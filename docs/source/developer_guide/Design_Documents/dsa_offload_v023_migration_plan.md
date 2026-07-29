# DSA 稀疏卸载 v0.23 迁移计划

> - 最后更新：2026-07-29
> - 当前阶段：P0-P5 已接通并完成 GLM-5.1 910C 初验；P6 已复用原生
>   FULL decode capture/replay 完成框架实现，等待服务器图编译与精度验证
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

- 当前 LIDU 算子 ABI 只支持上述三档 resident budget；caller-owned
  输出张量的固定列宽为 `16384`。resident budget 与输出容量不是同一概念。
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

v0.23 的这次重构把三种此前容易混在一起的语义分开了：

1. spec/registry 决定单层 cache 的字节布局和块生命周期算法；
2. group 决定哪些层共用一张逻辑 block table；
3. tensor 决定 worker 最终分配多少物理字节。

原生多 group coordinator 仍共享一个 `BlockPool`，是因为其 hybrid cache
解决的是“不同 attention 类型共享一个逻辑 block ID 空间”的问题。DSA
的 Indexer/MLA 不仅 page 字节不同，容量也按 ratio 不同，两边 block ID
都从 0 独立编号，因此不能复用该单 pool 假设。

P2 的实现将 `num_blocks` 保持为 MLA 基础容量，以配置 ratio
计算 Indexer tensor 容量；各物理池的最终 block 数由 finalized
tensor/spec 尺寸推导，避免重新引入 v0.19 的动态 `CacheConfig` 属性。
跨 rank 收敛时，vLLM 会按最小 `num_blocks` 等比例缩小每个
`KVCacheTensor.size`，因此 Indexer:MLA ratio 仍能保持不变。

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
2. 如果 request target budget 或 cache 布局确实无法从现有接口派生，只
   增加很薄的 Ascend 调度适配，并始终调用 `super()`；
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
| DCP/PCP | 均为 1 |
| pipeline parallel | 1 |
| KV-cache metrics/events | 关闭 |
| 图模式 | 原生 `FULL_DECODE_ONLY`；DSA row-mode gate 只放行单 token decode |

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
| P1 | 0 | 类型化配置与能力型模型识别 | GLM-5.1 服务器验证通过 |
| P2 | 0 | Indexer/MLA spec、字节规划、物理 tensor 解耦 | GLM-5.1 910C cache 初始化与容量报告通过；DeepSeek-V3.2 回归待补 |
| P3 | 0 | 独立 manager/coordinator 与请求 cache 布局合同 | Linux + Ascend 59 项累计 UT 通过 |
| P4 | 1 | `NPUInputBatch`、block table、统一行状态 | Linux + Ascend UT、GLM-5.1 cache-init/disabled 回归通过 |
| P5 | 1 | eager 数据面：dump、LIDU、KSC、SFA-Offload | GLM-5.1 910C、bsz=4、8K prompt eager 初验通过；完整精度矩阵待补 |
| P6 | 1 | 复用原生图捕获/replay | 框架与合同 UT 已实现；等待 910C capture/replay 验证 |
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
- GLM-5.1 W4A8、EP/TP16 服务器验证：

  1. DSA 关闭时正常拉起并生成；
  2. DSA 开启时类型化配置和 GLM-5.1 能力识别通过；
  3. `async_scheduling=True` 准确触发启动期拒绝；
  4. 最终 `block_size=128` 校验通过。

同一 `disabled` 配置在两个独立进程中已经观察到 decode token 分叉，因此
该环境不具备跨进程逐 token 确定性。P1 的非回归判断以“DSA 配置没有计算
路径消费者、关闭/开启均可正常拉起、同模式漂移已被基线复现”为依据。
DeepSeek-V3.2 的服务器回归可与 P2 cache 初始化测试合并执行。

## 10. P2：物理 cache 解耦

P2 只处理“空间是什么、占多少、如何绑定”，不提前迁移请求阶段和算子热
路径。

### 10.1 已实现设计

1. 在 `KVCacheSpecRegistry` 注册：

   - `DSAIndexerKVSpec -> DSAIndexerKVCacheManager`；
   - `DSAResidentMLAAttentionSpec -> DSAResidentMLAKVCacheManager`。

   两个 spec 各自作为 uniform-type base，禁止被自动归并。

2. DSA 模式下，`NPUModelRunner.get_kv_cache_spec()` 显式产生：

   - 全量 Indexer dense plane；
   - MLA resident plane；
   - 非 DSA 模式继续产生原生 packed `AscendMLAAttentionSpec`。

3. `IndexerWrapper` 在 DSA 拉起期保留原始 Indexer cache 的
   `static_forward_context` identity，并把原生 fp8-naive 132B 复合布局
   修正为算子实际消费的 128 维 bf16/fp16 单向量布局。
4. resident MLA spec 的 page size 只计算 MLA latent 与 RoPE，第三个
   `sparse_head_dim` 为 0，不再重复计算 Indexer K。
5. 分组固定为 Indexer 在前、resident MLA 在后；当前 GLM-5.1 和
   DeepSeek-V3.2 要求每个 resident 层都有独立 Indexer cache。
   `skip_topk` 只表示复用 top-k，并不表示省略 Indexer cache。显式共享
   Indexer 的更新模型暂不支持。
6. `KVCacheConfig.num_blocks` 表示 resident MLA base blocks；
   Indexer tensor 使用 `ratio * num_blocks`，两组最终容量均从
   `KVCacheTensor.size / page_size_bytes` 反推。
7. 自动容量按一个 MLA base block 的加权物理成本计算，显式
   `num_gpu_blocks_override` 仍沿用 base-block 语义。
8. model runner 分别分配并 reshape Indexer 4D tensor 与 MLA
   latent/RoPE tuple。二者在同一 transformer layer 下拥有不同 module
   name；vLLM 通用 `bind_kv_cache` 在昇腾上会拒绝这种同层双 cache，
   因此 DSA 使用仅在解耦模式生效的类型化 binder，将两张 cache 分别
   绑定回各自 `static_forward_context` 模块。非 DSA 路径继续原样调用
   上游 binder。
9. 只从 finalized `KVCacheTensor.size`/spec 推导物理容量；不在
   `CacheConfig` 增加 `dsa_num_blocks` 一类影子字段。
10. 容量报告中的总字节、两组 blocks 与 tokens 均来自最终 tensor。
11. 两个 plane 继续各自拥有 `MultiGroupBlockTable` 中的 block table 和
    slot mapping，但只为真实 resident MLA attention 构建一份 SFA
    metadata。Indexer 的两个寻址 tensor 作为该 metadata 的附加视图传递，
    不复制数据，也不创建第二个 SFA metadata builder。
12. Indexer 4D tensor 直接按 `DSAIndexerKVSpec` reshape；不再为了取得
    shape 而把 Indexer cache group 伪装成 `AscendSFABackend`。

### 10.2 当前边界

- P2 改变了物理 cache ABI；P5 已在 DSA 开启时把独立 Indexer tensor
  重新连接到 resident SFA forward；
- 非 DSA 路径仍按原生 packed tuple 消费 cache；
- P5 已在 GLM-5.1 910C 完成自定义算子编译和 bsz=4、约 8K prompt
  eager 生成初验；完整数据集精度、性能和 DeepSeek-V3.2 回归仍待补；
- Indexer group 是“有独立物理 cache/block table、无独立 attention
  forward”的寻址 plane。`attn_groups[indexer_gid]` 有意为空；model runner
  对该 group 单独完成 kernel block-size 选择与 4D tensor reshape，并把
  寻址视图附着到 resident SFA metadata。P4/P5 继续扩展同一份 common/SFA
  metadata，没有恢复第二套 builder；
- DCP/PCP、pipeline parallel 以及 KV-cache metrics/events 首版启动期拒绝。
  PP 的全局 group 投影允许 stage 出现空 group，当前尚未定义空平面的
  双 pool 容量语义；metrics/event payload 则以裸 block ID 为键，而
  两个独立 pool 都从 0 编号，直接共用会发生碰撞。

### 10.3 P2 禁止事项

- 只恢复 `DeepseekV32IndexerCache.get_kv_cache_spec()` 而不修改 consumer；
- 在 `AscendMLAAttentionSpec.page_size_bytes` 中同时计算两套 Indexer 空间；
- 用同一个标量 block 数假装两个物理池容量相同；
- 为了快速跑通而复制 v0.19 `patch_kv_cache_utils.py`。

### 10.4 P2 验收门槛

- DSA 关闭时原生 packed layout 完全不变；
- DSA 开启时不存在 Indexer 双重分配；
- tensor shape、page bytes、总分配字节和报告四者一致；
- TP 各 rank 的 group 顺序与容量一致；
- GLM-5.1 和 DeepSeek-V3.2 均通过 cache 初始化；
- 仅完成空间初始化尚不能宣称 DSA 推理可用。

### 10.5 910C 验收入口

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode cache-init \
  --result-json /tmp/dsa-p2-glm51-cache-init.json
```

DeepSeek-V3.2 使用相同命令替换模型路径。预期：

1. 权重和 KV cache 初始化完成，脚本不进入 `generate`；
2. 日志恰好出现一份 `DSA HBM CACHE CAPACITY REPORT`；
3. `Indexer dense plane blocks == MLA resident plane blocks * ratio`；
4. `KVCacheConfig.num_blocks` 对应 resident MLA，而非两组块数之和；
5. 没有 `Some layers are not correctly initialized`、group 数量或
   tensor page 对齐错误。

### 10.6 当前验收结果

2026-07-28 在 Ascend 910C、GLM-5.1 W4A8、TP16/EP 环境完成：

- `test_config.py`、`test_model_support.py` 和 `test_kv_cache.py` 共
  41 项单测全部通过；
- DSA 关闭时原生 packed cache 路径正常完成生成；
- DSA `cache-init` 模式完成模型、双平面 KV cache 和 worker 绑定初始化，
  并按设计跳过 `generate`；
- `async_scheduling=True` 在权重加载前准确命中启动期拒绝；
- 未观察到 group 数量、同层双 cache 绑定、tensor reshape 或 page 对齐
  异常；
- 最终容量报告给出 resident MLA 1,536 blocks/196,608 tokens、Indexer
  4,608 blocks/589,824 tokens，严格满足 3:1 比例；最大 resident budget
  12,288 tokens 加一个 128-token 尾块后，每请求占用 12,416 resident
  slots，因此 MLA 容量上限为 15 个请求；Indexer 在
  `max_model_len=8,192` 时容量上限为 72 个请求；报告包含
  `Configured decode limit`，该配置最终由 `max_num_seqs=2` 限制为
  2 个请求；
- 最终两组 KV tensor 共分配 29,444,014,080 bytes（27.42 GiB）。

当前仍需补充一项证据，完成前不将 P2 标记为全部验收完成：

1. 待测试机具备权重后，完成 DeepSeek-V3.2 的 `cache-init` 强制回归。

本轮 `cache-init` 成功只证明控制面、物理初始化和 worker 绑定成立；P5
已经另用长请求 eager 生成覆盖真实 dump、LIDU/KSC/SFA-Offload。两类证据
不能互相替代。

### 10.7 与 v0.19 语义实现的复核结论

本轮不是按文件逐段搬运 v0.19 patch，而是按其功能合同逐项映射到 v0.23：

| v0.19 已验证合同 | v0.23 当前实现 | 结论 |
|---|---|---|
| Indexer 与 MLA 使用不同 spec identity | 两个 DSA 专用 spec 注册到 `KVCacheSpecRegistry` | 已对齐 |
| MLA page 不重复包含 Indexer 字节 | resident spec 的第三段 `sparse_head_dim` 为 0 | 已对齐 |
| Indexer 容量为 MLA base blocks 的 ratio 倍 | ratio 编码进 finalized `KVCacheTensor.size` | 已对齐，且移除动态影子字段 |
| 两个 plane 使用独立 block ID 空间 | 每个 group 拥有独立 `BlockPool` | 已对齐 |
| worker 分别分配、reshape 和绑定两张 cache | Indexer 4D view、MLA tuple 和窄绑定器均已初始化通过 | 已对齐 |
| attention 共用一次 forward，但能取得两组寻址信息 | resident SFA metadata 附带 Indexer block table/slot mapping view | 已接入 P5 算子消费 |
| Indexer 保留完整上下文，MLA 在 ENTER 后收缩为 budget+tail | planner、双 manager 和 worker ENTER 全量表投影已经接通 | P4 结构与服务器初验已对齐 |
| 满块 dump、DRAM ledger、LIDU/KSC/SFA-Offload | 固定 DRAM arena、统一 runtime 和 attention 算子链已接通 | P5 框架完成，910C 运行验证待补 |

因此，“Indexer/MLA 初步解耦完成”只用于描述 P2 的物理空间与初始化
合同。它不等价于请求运行期解耦完成，也不等价于 DSA 稀疏卸载已经可生成。
P3 已直接建立在现有类型化 manager 和双 pool 上，不需要退回 v0.19 的
全局 scheduler/manager patch 形态。

## 11. P3：分配与请求 cache 布局计划

### 11.1 最终采用的结构

- 使用 vLLM-Ascend coordinator factory 创建 DSA coordinator；
- Indexer manager 维护完整上下文 block；
- MLA resident manager 维护 sparse budget、保留尾块和阶段转换所需空间；
- admission 进行 component-wise 容量检查，不能只看某个合并比例；
- target resident budget 在请求首次成功 admission 时冻结；
- 使用 `scheduler_config.scheduler_cls` 安装薄
  `DSAOffloadScheduler`，只表达 DSA 阶段屏障和输出后释放时机；
- 所有通用 token budget、waiting/running 队列处理、请求 admission 和
  `SchedulerOutput` 构造仍调用 vLLM 原生 `Scheduler.schedule()`。

### 11.2 请求 cache 布局语义真源

`DSARequestCachePlanner` 是 scheduler/core 侧唯一 DSA cache 布局账本。
它不接管 vLLM `RequestStatus`，不修改 vLLM `Request`，也不在 worker 各
rank 重新推导状态。每个请求仅持久化一个 slotted state：

- 当前 `PREFILL/DENSE_DECODE/ENTER_SPARSE_DECODE/SPARSE_DECODE`；
- 按 prompt token 数选择且跨 decode step 冻结的 target resident budget；
- 当前 sparse budget 和 resident 有效 token 数；
- prefill resident 满块是否已经释放。

planner 使用轻量不可变 plan 和可变持久 state 组成的 `plan/commit`
两阶段协议：

1. `plan()` 只计算候选布局，不推进跨 step 状态；
2. manager 先分别检查 Indexer pool 与 resident pool；
3. 容量满足后修改两个物理 block table；
4. 所有分配成功后才 `commit()`，原地更新该请求唯一 state；
5. 容量失败返回 `None` 时，阶段和 resident 表保持原状。

每个 step 只创建一个 slotted plan。`tail_tokens` 与 ENTER 的 resident
整表替换标志由 plan 属性派生，不单独存储；旧实现中未被生产路径消费的
`resident_tokens_need_slot` 已删除。这样既保留失败原子性，也避免 steady
decode 反复创建初始状态、下一状态及普通 dataclass `__dict__`。

四阶段的物理语义如下：

| 阶段 | Indexer plane | resident MLA plane |
|---|---|---|
| `PREFILL` | 按完整 prompt 分配 | 按完整 prompt 分配，供 prefill 与 dump |
| `DENSE_DECODE` | 随完整上下文增长 | 随完整上下文增长 |
| `ENTER_SPARSE_DECODE` | 保留并继续增长完整上下文 | 一次性替换为 `budget + tail`；旧尾块未满时原块复用 |
| `SPARSE_DECODE` | 随完整上下文增长 | 物理块表固定，只有有效尾长随 step 变化 |

当 prompt 恰好块对齐时，首个 decode token 需要一张新的尾块；prompt
未块对齐时，ENTER 保留原 dense-prefill 尾块，避免丢失其中已有 KV。

### 11.3 薄 Scheduler 适配

P3 证明仅靠 manager 无法完整表达以下两个时序约束，因此采用子类扩展点，
而不是恢复 v0.19 的全局 monkey patch：

1. 首版禁止 prefill/decode 进入同一个 model forward。薄 scheduler 在调用
   `super().schedule()` 前临时隐藏不属于本轮 phase 的队列视图，返回后按
   原顺序恢复；它不复制上游调度循环。
2. prefill 满块只有在对应 model forward 已返回后才允许释放。当前首版
   dump 与模型执行位于同一 NPU stream，stream 内保序，因此
   `update_from_output()` 返回点可以作为同步边界。未来引入异步多 stream
   dump 时，必须增加 event/readiness 协议，不能沿用该假设。

waiting prefill 只有在 **两个物理 pool 都能容纳完整 dense prompt** 时才
阻塞已有 decode；否则先让 decode 前进并释放资源，避免 waiting 请求导致
全局停滞。动态 preemption/resume 尚无 DRAM ledger 恢复合同，首版一旦
触发即显式 `RuntimeError`。

纯 steady decode 且 waiting/skipped-waiting 均为空时，DSA scheduler
直接调用上游快路径，不执行 phase gate 扫描，也不构造临时空队列。

`enable_chunked_prefill=True` 和非零
`long_prefill_token_threshold` 都会在启动期拒绝，防止 v0.23 通过后者
隐式切分长 prompt。

### 11.4 P3→P4 传输合同

coordinator 中的两张 block table 是 scheduler/core 侧逻辑真源。v0.23
原生 cached-request 输出只携带各 group **新增** 的 block IDs；ENTER
需要把 worker 的 resident 表从 dense 全表替换为 `budget + tail`，不能用
“追加新块”正确表达。

P4 使用 `DSAOffloadSchedulerOutput` 薄子类增加单个类型化 projection
字段。它按 dataclass 固定字段浅包装基线输出，不复制调度输出构造逻辑，
不修改 vLLM 类，也不动态挂四份字典。projection 携带已 commit 的阶段、
resident 有效长度、冻结 budget，以及仅 ENTER 行需要的全量 resident
replacement。

worker 先完整执行原生 `_update_states()`，让 remove/add/condense/backend
reorder 得到最终行序。steady 行序与 projection 一致时，四个 scheduler
语义列直接批量写入；只有请求集合或行序变化时，才通过一次 request-id
映射同步 resident pool/DRAM ledger 行。这样既不复制基线请求生命周期，
也不会把 scheduler 的传输顺序误当成 worker 行号。

scheduler projection 仍需要一次 O(B) 遍历：只有 `super().schedule()` 返回
后才能知道最终 scheduled request 集合，而 vLLM v0.23 在输出构造处没有
逐请求扩展钩子。把它强行合并进基线循环需要复制整个 `schedule()` 或修改
vLLM，均违反本迁移原则。worker steady decode 不再额外做 DSA request-id
查找循环。

### 11.5 P3 验收门槛

- add、decode grow、ENTER、free 的两组 block 账本一致；
- prefill 后释放只保留未满尾块；块对齐 prompt 不误保留旧满块；
- 容量失败不提前推进阶段，不留下半替换 resident 表；
- waiting prefill 容量不足时不饿死已有 decode；
- 首版不支持 preemption 时必须显式拒绝；
- 不复制或全局替换 SchedulerOutput；
- 不复制调度循环；
- 不在 worker 各 rank 独立决定逻辑 block 分配。

P3 已在 Linux + Ascend 环境与 P0-P2 用例一并验证：共收集并通过 59 项，
无 skip、xfail 或失败。其中新增覆盖包括请求 budget 冻结、四阶段转换、
双 pool 分配与释放、ENTER 容量失败原子性、块对齐尾块，以及薄 scheduler
的 phase barrier、双 pool admission 和无 waiting 请求快路径。测试命令为：

```bash
python -m pytest \
  tests/ut/dsa_offload/test_config.py \
  tests/ut/dsa_offload/test_model_support.py \
  tests/ut/dsa_offload/test_kv_cache.py \
  tests/ut/dsa_offload/test_request_cache_layout.py \
  tests/ut/dsa_offload/test_kv_cache_layout.py \
  tests/ut/dsa_offload/test_scheduler.py \
  -vv --tb=short
```

## 12. P4-P7 摘要

### P4：统一行状态

已实现：

1. `DSARequestCacheLayoutProjection` 用紧凑 tuple 列承载四个 scheduler
   标量合同，
   只有 ENTER 行传全量 resident block IDs；
2. `DSAOffloadSchedulerOutput` 是 pickle-safe 的薄子类，所有基线字段做
   O(1) 浅包装；
3. `NPUInputBatch` 仅在 DSA 开启时创建一个 `[6, max_num_reqs]` 的
   `CpuGpuBuffer`：四个投影列，加派生的 `row_mode` 与稳定
   `resident_pool_index`；六个 device 列均保持连续；
4. worker 在基线最终行重排之后刷新 active-prefix；stable 行序批量复制
   四个投影列，结构变化时一次性同步 pool 行；未来 graph
   captured-prefix 的额外行统一写 PAD；
5. ENTER 覆盖 worker request resident 账本及对应 `BlockTable` 行，
   DENSE/SPARSE 不产生额外块表写入；
6. P4 不执行 H2D。P5/P6 复用同一 owner，一次复制后供 eager/graph
   消费；
7. PAD 容量在 owner 初始化时一次设置，batch 缩小时仅清退旧 active
   tail，不在 steady step 重写 `[active_rows:max_num_reqs]`。

projection pickle、乱序行映射、ENTER 整表替换、缺失替换拒绝、PAD
初始化和 stable 行批量刷新均已纳入累计 UT，并已在 Linux + Ascend
环境通过。

P4 服务器累计回归命令如下：

```bash
python -m pytest \
  tests/ut/dsa_offload/test_config.py \
  tests/ut/dsa_offload/test_model_support.py \
  tests/ut/dsa_offload/test_kv_cache.py \
  tests/ut/dsa_offload/test_request_cache_layout.py \
  tests/ut/dsa_offload/test_kv_cache_layout.py \
  tests/ut/dsa_offload/test_scheduler.py \
  tests/ut/dsa_offload/test_scheduler_output.py \
  tests/ut/dsa_offload/test_input_batch.py \
  -vv --tb=short
```

### P5：eager 数据面

当前实现：

1. `DSAResidentTokenPool` 为每个活跃请求分配稳定 pool row，并为每层持有
   LIDU 原址更新的 `cache_slots`；额外一行预留给 PAD；
2. `DSAHotDRAMStore` 在 worker 初始化后使用固定容量、固定地址的逐层
   NOPE/ROPE swapped-memory arena。逻辑表按
   `pool row × logical full block` 管理，block 0 为空映射；
3. `DSAOffloadRuntime` 持有 eager/后续 graph 共用的 row metadata、
   active DRAM table、满块 dump 列和 caller-owned LIDU scratch；
4. Indexer 与 resident slot mapping 继续复用
   `MultiGroupBlockTable/BlockTable.compute_slot_mapping()`，只传入不同的
   position view，不建立第二套 block-table 实现；
5. prefill 复用基线 lightning-indexer，但读独立 Indexer plane；
6. decode 按整 batch 执行 LIDU→KSC→SFA-Offload，DENSE/ENTER/SPARSE
   不在 Python 侧拆子 batch；
7. SFA 完成后，同 stream 调用通用 full-block dump 算子。scheduler 只在
   model output 返回后释放 prefill resident 满块；
8. LIDU topK/slot/miss/tail 输出是逐层短生命周期 scratch，所有层串行
   复用同一固定地址；只有 `cache_slots` 是逐层跨 step 持久状态；
9. 每 forward 的满块边界判定使用预分配 NumPy scratch。无边界的 steady
   step 不构造 dump jobs，也不刷新 DRAM table H2D；
10. request release 只回收 pool row，下一次 acquire 在复用前完成唯一一次
    全层状态清理，避免对大 `cache_slots` 行重复写两次。

当前限制：

- P6 已在同一 owner 上增加 captured-prefix + PAD；P5 本身仍保持
  active-prefix，不建立 graph-only 状态；
- 原生 `IndexCache/skip_topk` 与逐层 LIDU resident 状态语义不同，启动绑定
  时显式拒绝；
- DRAM arena 不扩容；preemption、prefix cache 与跨请求 block 共享留到 P7；
- 需要在 910C 完成四个自定义算子的编译、专项 UT 和长请求生成验证。

P5 服务器累计 UT 入口：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short
```

其中 `test_resident_pool.py`、`test_dram_store.py` 和 `test_runtime.py`
分别覆盖稳定行/逐层状态、DRAM 预留与整行释放、LIDU scratch 复用及 dump
计划幂等性。通过 UT 后仍需执行自定义算子编译和 GLM-5.1 长请求 eager
生成；短 prompt 或 `cache-init` 不能覆盖 P5。

2026-07-29 的 GLM-5.1 W4A8、910C、TP16/EP、bsz=4、约 8K prompt
eager 对照中，自定义算子链完成生成。修复 persistent InputBatch 临时移除
请求时误释放 resident pool/DRAM ledger 后，四行输出均保持连贯，未再出现
单行异常提前 EOS；该结果是 P5 冒烟通过，不替代 QA 数据集精度回归。

### P6：图模式

P6 复用 v0.23 原生 FULL decode capture/replay。没有新增 dispatcher，也
没有复制 `_model_forward` 或 attention metadata builder。实现分为四层：

1. `graph_gate.py` 只读取 P4 已投影的 InputBatch 阶段：

   - 单 token `DENSE/ENTER/SPARSE` 任意混排允许 FULL graph；
   - prefill、multi-token 和 capture-size miss 是正常 eager；
   - owner 缺失、行数错位是内部合同破坏，直接报错。

2. 原生 dispatcher 的最终 uniform FULL keys 是图容量唯一真源，仍负责
   选择 capture size、向上 padding 和 DP 图模式协调：

   - 用户原始 capture-size 会被 `max_num_seqs` 和 attention backend 能力
     过滤，DSA gate 不读取这份原始列表；
   - 当前只支持具有独立 decode routine 的模式，例如
     `FULL_DECODE_ONLY`，拒绝 mixed/decode 共用的精确 `FULL`；
   - DP replica 本来允许处于不同阶段。任一 replica 需要正常 eager 时，
     所有 rank 跟随原生全局决议共同 eager；DP=1 且没有 cascade/encoder
     等动态 blocker 时，gate 允许却未选择 FULL 才 fail-fast；
   - DSA graph 开关关闭时要求原生 `cudagraph_mode=NONE`，避免启动期仍
     capture 一组永远不会使用、且缺少 DSA dummy metadata 的基线图。
3. eager 与 graph 共用以下 owner：

   - `DSAInputBatchCacheLayout`：eager 取 active-prefix，graph 取
     captured-prefix，尾行保持 PAD；
   - `DSAResidentTokenPool`：真实请求按稳定 pool row 跨 step 持久化；
   - `DSAOffloadRuntime`：active DRAM table、LIDU scratch 和 dump 列地址
     不变。graph dump 采用 captured-row 固定宽度，未使用行写
     `src=0, dst=-1` 空转；eager 仍只提交紧凑 jobs。

4. 原生 dummy warmup/capture 不经过 `_prepare_inputs()`。P6 只在 dummy
   生命周期内把上述 owner 的 device-prefix 临时设置为合法 SPARSE
   first-fill，完整覆盖 attention metadata 构造和 model forward，随后在
   `finally` 中恢复。CPU 请求语义真源不被 dummy 修改，LIDU 原址刷新过的
   capture 行也会被清零，不能污染首个真实请求。

真实 replay 的顺序保持基线范式：

```text
_update_states
  -> _prepare_inputs（刷新统一 active owner）
  -> DSA gate
  -> 原生 dispatcher / padding
  -> finalize execution view
  -> 原生 _build_attention_metadata
  -> 原生 FULL graph replay
```

P6 新增 UT 覆盖 DENSE/ENTER/SPARSE gate、正常 eager 原因、状态错位拒绝、
最终图配置合同、多 capture size 共用固定 owner、capture dummy 恢复、
PAD resident 行，以及 dump `dst=-1` 固定宽度空转。
Windows 本机缺少已编译的 `vllm_ascend._build_info`，只能完成
`py_compile`、Ruff 与 diff 检查；完整 UT 和 ACL graph 必须在 Ascend
安装环境执行。

服务器建议按以下顺序验证：

```bash
python -m pytest tests/ut/dsa_offload -vv --tb=short

python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode eager \
  --max-num-seqs 4 \
  --result-json /tmp/dsa-p6-eager.json

python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode graph \
  --max-num-seqs 4 \
  --result-json /tmp/dsa-p6-graph.json
```

随后用长短混合 prompt 覆盖全 DENSE、全 SPARSE、DENSE/ENTER/SPARSE
混排、新满块 dump、active 3→captured 4 PAD 和请求结束后行复用。

### P7：扩展

依次评估 chunked prefill、prefill/decode mixed、preemption、prefix cache、
MTP、async scheduling、KV transfer 和 A5 算子实现。

## 13. v0.19 机制迁移映射

| v0.19 机制 | v0.23 目标 |
|---|---|
| 动态 `CacheConfig` 属性 | `AscendConfig.dsa_offload_config` |
| 架构名白名单 | 模型能力协议 |
| 全局 Scheduler monkey patch | coordinator/manager 与薄 DSA 调度适配 |
| 全局替换 SchedulerOutput 类 | 只增加一个 projection 的薄 Ascend 子类 |
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
| 是否需要薄 DSA Scheduler 适配 | P3 | 已确认需要；只保留 phase barrier、输出后释放和 preemption 拒绝 |
| DSA 状态是否进入 SchedulerOutput | P3/P4 | 已采用一个类型化 projection 的薄子类；不修改或全局替换 vLLM 类 |
| preempt 后 tokenwise row 与 DRAM ledger 生命周期 | P7 | row 释放，DRAM ledger 独立保留 |
| prefix/content hash 的 DRAM 身份 | P7 | 首版使用 request lifetime + logical block |
| A5 算子实现 | P7 | 框架 ABI 保持不变，由算子侧适配 |

## 16. 变更记录

- 2026-07-29：

  - 完成 GLM-5.1 W4A8、910C、TP16/EP、bsz=4 的 P5 eager 长请求初验；
  - 修复 persistent InputBatch 临时移除未调度请求时错误释放请求级
    resident pool 与 DRAM ledger 的生命周期问题；
  - P6 增加纯 DSA row-mode gate，并复用原生 dispatcher、FULL graph
    capture/replay 和 attention metadata builder；
  - DSA 图容量改为读取原生 dispatcher 最终 uniform FULL keys，拒绝共享
    mixed-batch 的精确 `FULL`，并遵循原生 DP 全局 eager 决议；
  - `DSAInputBatchCacheLayout`、`DSAResidentTokenPool` 与
    `DSAOffloadRuntime` 增加 capture-only device-prefix 安装/恢复，
    不修改 CPU 请求语义真源；
  - eager 继续使用 active-prefix 与紧凑 dump jobs；graph 使用
    captured-prefix + PAD，并用 `dst=-1` 固定宽度空转 dump；
  - 增加 P6 gate、dummy 恢复、PAD 与 dump execution-view UT，并更新
    demo 的 `graph` 服务器入口。

- 2026-07-28：

  - 创建 v0.23 迁移计划；
  - 冻结首版支持矩阵和 P0 回归合同；
  - 完成 P1 类型化配置、模型能力识别和两阶段 block-size 校验；
  - 增加 P1 disabled/enabled/reject-async 三模式服务器 smoke demo；
  - 完成 GLM-5.1 W4A8、EP/TP16 的 P1 服务器验证，并记录原生路径跨进程
    decode 非确定性；
  - 完成 DSA Indexer/resident MLA 独立 spec、group、tensor 和容量规划；
  - 完成双物理 `BlockPool`、类型化 manager 与逐 component admission
    基础；
  - 使用与 scheduler 导入时序无关的原生 `KVCacheManager` 条件包装，
    消除早期平台 patch 静默漏装 DSA admission 的风险；
  - 增加同层 resident MLA/Indexer 双 cache 的窄绑定器，绕开上游
    非 CUDA 平台通用 binder 对同层多 cache 的显式拒绝；
  - 增加跨 rank tensor 等比收缩、plane ratio、weighted admission 和
    双 pool reset 单测；
  - 将服务器 smoke 更新为 P2 disabled/cache-init/reject-async，明确
    P4/P5 接通前禁止把生成结果作为 DSA 正确性证据；
  - 完成 GLM-5.1 W4A8、TP16/EP 的 P2 服务器初验：41 项 DSA 单测、
    disabled 生成、cache-init、reject-async 和双平面容量报告均通过；
    DeepSeek-V3.2 回归保留为待补证据；
  - 对照 v0.19 的完整功能合同复核 v0.23 P2：spec、page bytes、finalized
    tensor、双 pool、worker reshape/bind 均已对齐；将 DENSE/ENTER/SPARSE
    cache 布局转换和算子消费明确保留在 P3-P5；
  - 明确 DCP/PCP、pipeline parallel、KV-cache metrics/events 和显式
    shared Indexer 的首版边界。
  - 增加类型化请求 cache 布局 planner，以 plan/commit 协议冻结 prompt
    budget，并实现 PREFILL、DENSE、ENTER、SPARSE 的双平面分配；
  - ENTER 保留未满 dense 尾块，块对齐 prompt 为首个 decode token
    分配新尾块；steady sparse 只增长 Indexer，resident 物理表保持固定；
  - 增加基于 `scheduler_cls` 的薄 DSA scheduler，继续调用上游
    `schedule()`，仅提供 prefill-first phase barrier、输出后 resident
    释放和不支持 preemption 的显式错误；
  - 明确 v0.23 cached-request 的 append-only block delta 无法表达 ENTER
    resident 整表替换，该 worker 投影留给 P4；
  - 补充隐式 `long_prefill_token_threshold` 拒绝以及 cache 布局、容量失败
    原子性、队列恢复单测。
  - 将宽泛的 request lifecycle 命名收敛为 request cache layout，明确不
    接管 vLLM `RequestStatus`；
  - 请求持久 state 与每 step plan 改为 slotted 结构，steady decode 原地
    更新唯一 state；删除未消费字段和无效初始状态分配；
  - 新增 `dsa_offload_design.md`，持续记录当前已落地架构；迁移过程、风险
    与验收证据继续保留在本文。
  - P3 与既有 P0-P2 用例在 Linux + Ascend 环境累计 59 项全部通过；
  - 完成 P4 scheduler→worker 列式 projection；使用薄、pickle-safe 的
    `DSAOffloadSchedulerOutput`，不复制或 monkey patch 上游输出类；
  - 在原生 `_update_states()` 完成请求增删、压缩和 backend reorder 后，
    将四个 scheduler 语义列映射到 `NPUInputBatch` 最终行序；
  - 新增六列 SoA 固定容量 `CpuGpuBuffer`，额外派生 `row_mode` 与稳定
    `resident_pool_index`，为 eager active-prefix 与 graph
    captured-prefix + PAD 预留同一 owner；
  - ENTER 使用 scheduler 真源覆盖 worker resident request/block-table
    行，补齐 v0.23 cached-request append-only delta 无法表达整表替换的
    缺口；P4 累计 UT、cache-init 与 disabled 回归通过。
  - P5 新增稳定 resident token pool、固定 DRAM arena、统一
    `DSAOffloadRuntime` 和逐层 attention context；
  - 独立 Indexer slot mapping、resident slot mapping 与同一份 SFA
    metadata 接通，decode 采用整 batch LIDU→KSC→SFA-Offload；
  - prefill/decode 新满块统一在 SFA 后通过通用 full-block dump 算子写入
    DRAM；同 stream 保序，不维护无消费方的异步完成状态；
  - steady dump 边界判定和 LIDU 输出改为预分配 scratch，避免逐 step
    NumPy/tensor 临时分配；P5 等待 910C 编译和端到端验证。
