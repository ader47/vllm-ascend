# Delayed Nano Tail D2H Adaptation

## 状态与目标

本文描述 Nano 双尾块的延迟异步 D2H 方案。实现基线为：

```text
9b5c8fbee36c5e7c29a34461c39dbb83aade3d71
fix(pd): join compute stream once after D2D recv
```

新方案不移植 `78e36161cd` 的图内逐层 candidate/confirmed D2H，也不在拒绝采样后
创建一套新的 `committed_length` 或 PENDING 计划。它直接复用模型执行已有的
`num_computed_tokens`：某块在第 N 轮写满后，第 N+1 轮输入准备完成时，根据已经更新
好的 `num_computed_tokens` 判断该块是否需要卸载，并在图外启动异步 D2H。

目标如下：

- D2H 不进入 Target 或 MTP ACL graph。
- 满块 D2H 与下一轮完整模型图重叠。
- D2H 期间旧满块继续作为 dense tail 参与 Attention。
- 新 KV 只写另一个 tail page，复制源页不会被覆盖。
- D2H 完成前不推进 `stable_prefix`，也不允许 Host 消费该块。
- 普通 Decode 和 MTP 使用同一个 `num_computed_tokens` 判断，不增加第二套长度。
- CPU 只生成固定容量的请求级复制计划；全层 K/V 描述符在专用 D2H stream 上展开。
- 不增加 Python 后台线程；NPU stream 和 event 负责异步提交、跨流依赖及完成顺序。
- Target/MTP graph 不增加卸载、描述符或同步操作；融合 KV 写入替换原有分离写入链路。
- 支持 ACL graph、graph padding、请求槽位复用、PD、preemption 和 delayed-free。
- 第一版只保留一批 INFLIGHT D2H，不增加 PENDING 队列、第三个 tail page 或后台线程。

## 为什么从 `9b5c8fbee3` 开始

当前 `d2h` 分支在该基线之后叠加了以下旧方案：

```text
KV 写入本层 tail
    ↓
本层立即提交 candidate/confirmed D2H
    ↓
D2H 与本层 Attention 和后续层计算重叠
    ↓
图末 join_nano_d2h
```

新方案替换整段时序。从 `9b5c8fbee3` 开始可以保留原始 Nano ring、LIM 和 SFA 数据
路径，同时避免先引入再删除旧的图内 D2H 状态机。

该基线仍包含逐 token D2H、`index_copy_` 尾块写入，以及若干后来修复的 MTP/PD
问题，因此不能直接作为最终实现，仍需选择性移植后续修复。

## 提交移植矩阵

### 可以直接移植

| 提交 | 内容 | 处理 |
| --- | --- | --- |
| `f3dc9df3c4` | Nano PD Decode 禁用本地 prefix cache | 直接 cherry-pick |
| `021f2639a6` | unequal-TP 下避免重复拉取 Nano tail | 直接 cherry-pick |

### 只移植语义和测试

| 来源提交 | 保留内容 | 不移植内容 |
| --- | --- | --- |
| `78e36161cd` | `KvRmsNormRopeCache` 直接写 ring、完整 pool capacity、配置约束和 unsafe fallback 防护 | 图内逐层 D2H、candidate copy、graph join |
| `f7e6b82e19` | draft step 1+ 不构造无用恢复描述符 | 旧 flush metadata |
| `3206ac7b59` | capture/runtime 使用相同 MTP step 布局和实际 positions | 旧 D2H buffer 地址依赖 |
| `f6725d87ea` | TopK 复用时保留真实 tail 长度和可见性 | 旧 confirmed-plan |
| `ca4ef833b5` | PD rebase、generation 和 tail 所有权原则 | 旧逐层驻留水位实现 |
| `43dbfa3bc2` | 合法 pool slot 使用完整 Nano pool capacity | 旧 confirmed-plan 数组和读取流程 |

这些提交不能整笔 cherry-pick，否则会重新带入被替换的图内 D2H。适配时只保留上述
语义和对应回归用例。

## 两阶段实施

### 阶段一：能力移植

阶段一不改变卸载时序：

1. 移植 PD prefix-cache 和 unequal-TP 修复。
2. 移植 MTP capture/runtime metadata 布局、实际 positions 和 tail visibility。
3. 移植完整 Nano pool capacity、generation、padded request 和 unsafe fallback 防护。
4. 保留 Nano 配置约束：`block_size=128`；启用 speculative decoding 时仅支持 MTP、
   padded drafter batch 和每轮至少一个 draft token；Nano decode query width 为 1--7。
5. 保留非 Nano 路径原有 Host KV 更新。
6. 添加独立回归测试。

阶段一完成后仍沿用基线的逐 token D2H，必须可独立构建和运行。

### 阶段二：切换为延迟一轮的图外 D2H

以下切换必须在一个可运行提交中同时完成：

1. Nano Decode 使用 `KvRmsNormRopeCache` 直接写 HBM ring。
2. 删除 Nano 路径的逐 token `offload_new_kv`。
3. 每轮输入准备完成后，直接用现有 `num_computed_tokens` 检查是否出现新满块。
4. 图外构造固定容量的请求级计划；D2H stream 将其展开成固定形状的全层 K/V
   描述符并启动异步 D2H。
5. D2H stream 等待上一轮 compute stream 记录的 `source_ready`。
6. 下一轮边界等待上一批 D2H，完成跨 TP 通知后推进 `stable_prefix`。
7. 请求结束、PD 消费、preemption 和 slot 复用接入 delayed-free/flush 协议。
8. 删除逐层 Host→HBM tail restore、旧 graph join 和旧逐层 TP completion broadcast。

不能只删除逐 token D2H 而尚未接通满块 D2H，否则 Host KV 会缺失；也不能只切换
融合写入而继续无条件恢复 Host 尾块，否则旧 Host 内容会覆盖 HBM 中的新 tail。

## 目标时序

设 page A 在第 N 轮结束时成为完整块，page B 是下一写入页。

```text
第 N 轮 Target/MTP
    │
    ├─ KV 写入 page A
    ├─ 本轮 rejection 决定接受哪些 token
    ├─ 现有流程保存本轮 accepted-token count
    └─ 最后一个可能写 KV 的操作之后 record source_ready_N

第 N+1 轮图外准备
    │
    ├─ retire 更早的一批 INFLIGHT（如果有）
    ├─ 完成现有 input preparation
    │      └─ 现有逻辑用上一轮结果更新准确的 num_computed_tokens
    ├─ 判断 num_computed_tokens >= stable_prefix + 128
    ├─ 根据 block table、pool slot 和 generation 冻结 A 的请求级 INFLIGHT 计划
    ├─ 将小型计划 non-blocking H2D 到固定 device buffer
    ├─ D2H stream 展开全层 K/V 描述符
    ├─ D2H stream wait_event(source_ready_N)
    ├─ D2H stream 异步启动 D2H(A)
    └─ compute stream 启动 Target/MTP graph
            ├─ Attention 读取 A + B
            └─ 新 KV 只写 B

第 N+2 轮图外准备
    │
    ├─ TP0 retire 等待上一批 D2H(A) 的 Host-visible completion
    ├─ 仅当存在真实 INFLIGHT 时，TP0 将该批完成结果传播到所有 TP rank
    ├─ generation 匹配的请求在 CPU/device 两侧将 stable_prefix 前移 128
    ├─ 清除上一批 INFLIGHT
    └─ graph 中 A 转为 Host-backed history，dense tail 只读取 B
```

D2H 和 Attention 对 A 都是只读操作，因此可以并行。即使 D2H 在第 N+1 轮图执行
中途完成，也不动态改变本轮 metadata；A 在整轮中都参与 Attention，到第 N+2 轮边界
才转入 Host-backed history。

## `num_computed_tokens` 的唯一语义

本方案不定义 `committed_length`、`new_committed_len` 或其他平行长度。是否出现新满块
只读取已有的 `num_computed_tokens`：

```text
next_boundary = stable_prefix + block_size
ready = active
        and generation_matches
        and num_computed_tokens >= next_boundary
```

`num_computed_tokens` 表示已经执行过模型计算、因而在 KV cache 中有有效 K/V 的 token
数。sampler 刚产生但尚未进入下一次 forward 的 token 不计入该值。

MTP 不使用另一种长度。异步 speculative decode 已经根据上一轮
`valid_sampled_token_count` 修正 `num_computed_tokens`；卸载逻辑复用修正后的结果，不能
重复实现一套 rejection finalization，也不能使用 padded query width、optimistic shape
或 proposal 中间长度代替它。

例如：

```text
上一轮开始时 num_computed_tokens = 126
本轮计算 4 行，最终有效 2 行
下一轮 input preparation 后 num_computed_tokens = 128
128 >= stable_prefix(0) + 128，因此卸载 logical block 0
```

若最终只有 1 行有效，则下一轮 `num_computed_tokens = 127`，不启动 D2H。

实现必须保持以下边界：

```text
stable_prefix % 128 == 0
0 <= num_computed_tokens - stable_prefix < 256
```

query width 限制为 1--7，因此一个请求一轮最多跨过一个 128-token 边界。若检测到：

```text
num_computed_tokens >= stable_prefix + 2 * block_size
```

说明 retire、初始化或生命周期协议已失步，必须报错或进入冷路径恢复，不能静默跳过
一个块。

## CPU 侧满块规划

满块判断和请求级地址索引都是每请求几个整数运算，第一版放在 CPU/NumPy 上完成，
不新增 NPU planner kernel。planner 的复杂度必须是 `O(pool_capacity)`，不能在 CPU 上
按 `request × layer × K/V` 展开完整描述符。实际全层描述符展开和 HBM→Host 数据复制
在专用 NPU D2H stream 上异步执行。

CPU planner 复用现有请求 metadata：

- 当前请求行和 active mask；
- `num_computed_tokens` 的 CPU 视图；
- Nano pool slot、generation 和 block table；
- `stable_prefix`。

各层 K/V base address、token bytes 和 stride 在 manager 初始化时保存为静态 device
模板，只供 D2H stream 展开全层描述符，不进入 CPU planner 的逐轮循环。

上述请求状态必须具有随生命周期维护的 CPU 视图。planner 不得每轮从 NPU tensor
执行 `.item()`、`.cpu()` 或其他同步读回。CPU 和 device 控制状态在 slot 初始化、
generation 变化和 completion 消费时按同一协议更新。

异步 MTP 下，scheduler 的 CPU 值可能曾经按 draft 全接受乐观推进。planner 必须位于
现有 correction 之后，复用上一轮已经异步复制到 Host 的
`valid_sampled_token_count` 得到准确的 `num_computed_tokens`。这是现有
`num_computed_tokens` 更新流程的一部分，不创建新的长度状态，也不在卸载模块中重复
实现一套 MTP 语义。

CPU planner 只冻结一份固定容量的请求级计划：

```text
active / generation / source_slot / destination_slot / target_prefix
```

计划写入预分配的 pinned Host staging buffer，再在 D2H stream 上 non-blocking H2D 到
固定 device plan buffer。随后同一 D2H stream 使用预先保存的各层 K/V base address 和
stride 展开固定形状的全层描述符。因为 plan H2D、描述符展开和 D2H launch 位于同一个
stream，stream 顺序已经提供 descriptor-ready 依赖，不再增加冗余 event。

请求级计划一旦提交，在本次 plan H2D 消费完成前不得覆盖。正常路径先 retire 上一批
INFLIGHT 再复用唯一的一组 staging/device/descriptor buffer，因此不需要第二个计划 bank。
第一版不使用 `.item()`、逐请求 `.cpu()`、逐请求 Python 循环、动态 Python 队列或每轮
新分配 tensor。

这些 NPU 操作不需要新的 Python 后台线程。ModelRunner 主线程只按顺序向 D2H stream
提交命令，设备在该 stream 上异步执行；`wait_event` 只建立设备依赖，不阻塞提交线程。
AscendStore 的后台线程用于可能阻塞 CPU 的外部 Backend/网络调用，本地 HBM→pinned
Host DMA 不复用该线程模型。若复制 API 在 Host 侧等待完成，应改用真正的异步 DMA
接口，而不是用 Python thread 包装同步调用。

需要通过 profiling 检查读取上一轮 `valid_sampled_token_count` 的完成事件是否在稳态已
就绪。planner 的整数运算和冻结计划本身很小；热路径有两个有意保留的边界等待：读取
accepted count 前的 event wait，以及 retire active INFLIGHT 时的 D2H completion wait。
前者通常已被上一轮 draft preparation 掩盖，后者应被上一轮完整模型图掩盖；profiling
必须分别记录真正的剩余等待时间。若 accepted-count wait 实际成为瓶颈，再比较一个小型
融合设备 planner；在有数据前不引入自定义 kernel 或后台线程。

## Tail、prefix 和地址语义

```text
natural_prefix = floor(num_computed_tokens / 128) * 128
stable_prefix  = 已完成 D2H、可安全作为 Host-backed history 使用的边界
visible_len    = 当前 query row 可见的实际 KV 长度
```

正常稳态满足：

- 没有 INFLIGHT 时，`stable_prefix == natural_prefix`。
- A 正在 INFLIGHT 时，`stable_prefix == natural_prefix - 128`。
- `natural_prefix - stable_prefix` 最多为一个块。

传给 LIM/SFA 的 prefix 必须使用 `stable_prefix`：

```text
cache_tokens = min(stable_prefix, hot_topk_capacity)
logical_len  = cache_tokens + visible_len - stable_prefix
```

`cache_tokens` 表示 hot buffer 中可供 Attention 使用的历史行数，不等于算子一定启用
sparse TopK。copy-SFA 仍保留现有短前缀分支：

```text
0 < cache_tokens < configured_topk
    → attention_cache_tokens = 0
    → 通过现有 request-level dense-identity miss 路径准备历史行
    → Attention 对这段历史执行 dense 计算

cache_tokens >= configured_topk
    → attention_cache_tokens = cache_tokens
    → 使用 Indexer/TopK 选择的 sparse history
```

这一路径是原有 history cache fill，不是把 Host partial tail 逐层恢复到 ring，也不能重新
引入每层 tail-restore `ScatterElements`。因此第 N+1 轮的 Attention 集合为：

```text
已有历史（短前缀 dense identity 或长前缀 sparse TopK）
+ 正在 D2H 的满块 A
+ 当前尾块 B 的有效 token
```

Indexer 的 TopK 边界也使用同一个 `stable_prefix`。A 尚未 Host-ready，不能同时进入
Host-backed history（无论短前缀 dense 还是长前缀 sparse），否则会重复计算或读取尚未
完成的 Host 数据。

第 N+2 轮 completion 后，A 从 dense tail 集合移入 Host-backed history；它没有从
Attention 的逻辑 KV 集合中消失。此后 history cache fill 可以按现有规则把 A 的 dense
identity 行或 TopK 命中装入 hot buffer，而 ring 中只把 B 作为当前 dense tail。

物理地址规则保持不变：

```text
page = logical_block % 2
device_slot = pool_slot * request_stride + hot_tokens + logical_position % 256
```

只要 D2H(A) 未退休，A 不会再次成为写入页；新 KV 只写 B。因此两个 tail page 足够，
不需要第三个 page。

## 最小状态机

正常运行路径只有两态：

```text
IDLE
  └─ 下一轮根据 num_computed_tokens 发现新满块并提交 D2H → INFLIGHT

INFLIGHT
  ├─ D2H 完成且 generation 匹配 → 推进 stable_prefix → IDLE
  └─ generation 已失效 → 等待 DMA 完成但不推进新请求状态 → IDLE
```

D2H 提交或执行失败不进入新的可恢复状态；它是本轮服务的致命错误。实现必须阻止
`stable_prefix` 推进和相关资源复用，并让所有 TP rank 协同报错退出，不在第一版增加
重试、回滚或 `FAILED_INFLIGHT` 状态。

每个 pool slot 只保存：

| 状态 | 含义 |
| --- | --- |
| `generation` | 当前 slot 所属请求代次 |
| `stable_prefix` | 已 Host-ready 的块边界 |
| `inflight_active` | 当前批次是否包含该请求 |
| `inflight_generation` | 提交 DMA 时捕获的 owner 代次 |
| `inflight_target_prefix` | DMA 成功后允许推进到的边界 |

全层 K/V 的 source/destination/length 描述符属于 manager 级的一批 INFLIGHT buffer，
不是每请求动态队列。多个请求在同一轮填满时合并成一次固定形状、无效行长度为零的
D2H batch。

正常路径先 retire 上一批，再复用描述符启动新一批，所以不需要 PENDING bank、
`pending_snapshot_done`、奇偶 PENDING buffer 或 `finalize_nano_pending_blocks()`。
描述符只能在上一批 DMA 完成且其 completion 信息被消费后复用。

不同请求可以连续两轮产生满块：第 N+1 轮启动 R1，第 N+2 轮先退休 R1，再使用同一
描述符 buffer 启动 R2。同一请求因为每轮最多增加 7 个 token，不可能连续两轮各填满
一个新 128-token 块。

## 必要的跨流顺序

去掉 PENDING 不等于去掉 `source_ready`。CPU 在第 N+1 轮知道 A 已满，只能证明 A
逻辑有效；它不能证明第 N 轮 compute stream 已经完成对 A 的物理写入。这里的
`source_ready` 是整个 compute stream 的有序 frontier，不是每请求 event。

固定协议如下：

```text
第 N 轮 compute stream
    最后一个 KV writer 完成后 record source_ready_N

第 N+1 轮 CPU
    根据 num_computed_tokens 冻结请求级 INFLIGHT 计划

第 N+1 轮 D2H stream
    non-blocking H2D 请求级计划
    展开固定形状的全层 K/V 描述符
    wait_event(source_ready_N)
    sparse_copy / SDMA(A)
    record d2h_complete

第 N+2 轮 TP0 retire
    active INFLIGHT 执行 d2h_complete.synchronize() 或等价 Host-visible wait
    失败则禁止状态推进和资源复用，并进入跨 TP 协同报错退出

第 N+2 轮 CPU/communication
    仅对真实 INFLIGHT 批次传播 TP0 completion status
    各 rank 使用本地冻结的 generation/target_prefix 更新 CPU/device stable_prefix
```

`source_ready` 在完整 Target/MTP outer iteration 的最后一个可能写 KV 的操作之后记录。
MTP proposal 的多个 draft step 不是多个卸载 iteration，也不分别启动 D2H。

请求可能在块写满后暂时不被调度，直到更晚一轮才被 planner 发现。此时不保存或等待
已经轮转的原始 `source_ready_N`；D2H batch 等待“提交计划前最近一次真实 compute
iteration”的 source-ready frontier。因为所有 KV writer 位于同一有序 compute stream，
该 frontier 必然晚于这个请求最后一次写 A，因而是安全的保守 fence。请求缺席期间没有
新写入，不影响 source page 和 generation 的有效性。

`source_ready` 的 epoch 只对真实执行 KV writer 的 compute iteration 递增。无 forward 的
scheduler control tick 可以 retire 已有 INFLIGHT、参加所需 completion collective 或执行
terminal cleanup，但不能记录新的 `source_ready`、启动普通热路径计划或轮转 source-event
槽。capture/dummy invocation 同样不推进该 epoch。

## 多 TP 完成协议

Host pool 由 TP0 写入，但所有 TP rank 都必须在相同边界推进 `stable_prefix`。因此 TP0
D2H 完成后，需要把该批完成结果传播给其他 rank。每个 rank 已经根据同一份 scheduler
输入冻结相同的 `inflight_batch_active`、batch epoch、generation 和 target-prefix；正常路径
不重复广播整份请求级计划。

要求如下：

- TP=1 只建立本地 stream event 依赖。
- TP>1 只在存在真实 INFLIGHT 批次时执行一次 completion collective；没有 INFLIGHT 时
  不发起空 collective。
- 所有 rank 必须根据一致的 Host 侧 `inflight_batch_active` 和 batch epoch，以相同顺序参加
  同一批 completion collective。不能读取本地 NPU tensor 或 `event.query()` 后各自决定。
- TP0 必须先得到 Host-visible D2H completion，才能发布该批成功 status。
- completion status 直接通过 Host 控制通道广播 batch epoch；不要为了校验该标量新增
  NPU tensor 回读、`.item()` 或 completion device stream。
- collective 成功后，各 rank 使用本地冻结的 generation/target-prefix；迟到的旧完成仍需
  通过 generation 校验，不能推进复用 slot 的新请求。
- CPU/device `stable_prefix` 更新都排在 collective 完成之后。
- 同一批 completion 被所有 rank 消费前，不能覆盖相应控制 buffer。
- TP0 的 completion wait 或复制执行失败时，必须阻止状态推进，并通过本批原定的
  collective 或运行时错误传播机制让所有 rank 协同报错退出；不能只让 TP0 提前抛异常，
  也不尝试恢复服务。

各 rank 的 Host 批次状态一致是启用 TP>1 delayed Nano 的前置不变量，必须由实现断言和
测试覆盖。不能用“每个 outer iteration 固定一次 masked collective”掩盖状态分叉；若该
不变量无法保证，应明确拒绝对应配置。

## MTP 适配

MTP 与普通 Decode 使用同一个 `num_computed_tokens`，没有 MTP 专用的满块计划：

1. 现有 input preparation 根据上一轮 accepted-token count 修正
   `num_computed_tokens`。
2. CPU planner 在修正后判断是否跨过 `stable_prefix + 128`。
3. 整个本轮 Target/MTP proposal 固定使用本轮开始时的 `stable_prefix`。
4. step 1+ 使用实际 positions 计算有效 KV 长度，不使用包含 rejected/padded rows 的
   optimistic shape。
5. step 0 复用 TopK 时仍保留真实 tail 长度和可见性。
6. outer iteration 最后记录一个 `source_ready`。

拒绝 token 可以暂时留在 HBM 中并被后续写入覆盖，但它们不会增加修正后的
`num_computed_tokens`，因此不能触发满块 D2H，也不能推进 `stable_prefix`。

## ACL graph 适配

ACL graph 只包含 KV 写入、Indexer、sparse + tail Attention 和其他模型计算。D2H、
描述符 H2D、launch、retire、completion collective 和 event record 全部位于图外。

“不向 graph 增加卸载操作”不等于 graph 算子集合逐项不变。Nano Decode 的 KV 写入
需要用已有 `KvRmsNormRopeCache` 融合算子直接写 HBM ring，替换当前分离的
`RmsNorm + RoPE + index_copy_`；这是替换和减少操作，不是额外增加一段计算。原有
`npu_fused_copy_sfa_mtp` 继续读取 HBM block table 和 `logical_lens`，通过 metadata
直接看到 A+B，不增加一次 Attention、拼接或 tail copy。

目标图内路径为：

```text
KvRmsNormRopeCache 直接写 ring
    → Indexer / TopK
    → 原有 sparse history + dense A/B tail Attention
    → 其他原有模型计算
```

图内不得出现 Nano full-block `OffloadSparseCopy`、逐 token D2H、逐层 Host→HBM tail
restore、卸载 event wait 或 completion 更新。

- 删除旧 `join_nano_d2h()`。
- FULL graph 和 piecewise graph 使用同一 outer-iteration 协议。
- capture/runtime 使用相同 MTP step metadata 布局。
- capture/dummy run 不推进真实 iteration，不清理 INFLIGHT，也不执行真实 collective。
- runtime replay 结束后在图外记录 `source_ready`。
- 图中不创建或消费 Python 队列，不依赖动态图地址。

## Tail 恢复与 fallback

删除逐 token D2H 后，Host 中的 partial tail 不再持续更新。因此稳态 Nano 路径不能
保留原来的逐层 Host→HBM tail restore，否则 Host 旧内容会覆盖 HBM 中的新 KV。

仅允许以下恢复：

- 新 generation 第一次进入 Nano 时，从已确认 Host-ready 的边界做一次初始化恢复；
- PD connector 明确预载的 tail；
- preemption 恢复时，由 lifecycle 协议确认 Host 数据完整后恢复。

同一 generation 的 resident tail 不恢复。tail residency/restore mask 在共享 indexer
owner 处每请求更新一次，后续层复用，不能在每层产生 `ScatterElements`。

同一 generation 进入 Nano 后，不允许静默切换到依赖 Host partial tail 的非 Nano
fallback。图不适用时只能切换到具有相同 tail residency 语义的 eager Nano，或明确拒绝。

## 请求结束、抢占与 delayed-free

正常请求如果在下一轮仍在 batch 中，可以直接根据 `num_computed_tokens` 启动 D2H。
特殊情况是请求在 A 写满的同一轮结束或被抢占：它可能不会出现在下一轮正常 batch，
因此不能依赖热路径 planner 找到它。

这类冷路径使用相同数据源直接处理，而不是重新引入 PENDING 状态：

```text
finished/preempted request
    ├─ Host 不需要该新满块
    │      └─ 等待必要的旧 INFLIGHT 后直接释放，不启动新 D2H
    └─ Host/PD 后续需要该满块
           ├─ 根据最终 num_computed_tokens 构造一次性 flush 描述符
           ├─ 等待 source_ready
           ├─ 启动并等待 D2H
           └─ 完成后允许释放
```

connector 必须接入 `SupportsHMA.request_finished_all_groups()` delayed-free 协议：

1. 仍有 Nano binding 的结束请求保守返回 `True`。
2. 回执前保留 HBM source、Host destination、Nano slot、binding 和 generation。
3. worker 完成取消/flush/等待后返回 `finished_sending(request_id)`。
4. scheduler 收到回执后才释放 block 和 slot。

终止通知必须在 `_update_states()` 或其他可能重绑 block/slot 的操作之前被 worker 捕获，
但此时只标记请求进入 cleanup 并保留资源。随后先由统一的 `retire_nano_d2h()` 消费上一
批全局 INFLIGHT，再对仍未 Host-ready 的终止请求执行 cancel/flush；冷路径不能私自
消费同一批 INFLIGHT 后又让正常 retire 重复处理。

请求暂时未被调度不等于结束或抢占，不能因此释放 Nano slot。preemption/rebind 必须在
任何新请求写入该 slot 前等待旧 INFLIGHT；恢复时分配新 generation，并重新初始化
`stable_prefix`。旧 completion 即使迟到，也只能完成资源回收，不能推进新 generation。

第一版 delayed Nano 只在存在 delayed-free lifecycle provider 时启用。没有 provider
时必须明确拒绝，不能让 scheduler 立即复用 DMA 正在访问的资源。

## Host 可见性

`num_computed_tokens` 只能说明 HBM 中的数据有效，不能说明 Host 副本已经完成。Host
消费者必须依赖 `stable_prefix` 或 D2H completion：

- Attention/LIM 只把 `stable_prefix` 之前的数据当作 Host-backed history；
- 请求结束、PD handoff 或显式 Host 消费先执行 flush/等待；
- prefix cache 只有在能够等待 Host-ready 时才能发布新块。

Nano PD Decode 保持禁用本地 prefix cache，直到 scheduler 能按 Host-ready 状态约束
admission。非 PD/colocated Nano 也必须禁用本地 prefix cache，或接入等价的 completion
协议。

## 图外 API 建议

名字可以随实现调整，但职责保持简单：

```python
def retire_nano_d2h():
    """确认真实 INFLIGHT 已 Host-ready，按需传播完成并推进 stable_prefix。"""

def plan_and_launch_nano_d2h(num_computed_tokens, request_metadata):
    """冻结请求级计划，并在 D2H stream 展开描述符和启动满块 D2H。"""

def record_nano_source_ready():
    """在本轮最后一个 KV writer 之后记录 source-ready。"""

def flush_or_cancel_nano_requests(finished_or_preempted):
    """在冷路径保护资源并完成 delayed-free。"""
```

每个真实 outer iteration 的顺序为：

```text
0. 捕获 finished/preempted，标记 cleanup 并阻止相关 block/slot 复用
1. 统一 retire 上一批 INFLIGHT；仅有真实批次时跨 TP 推进 stable_prefix
2. 对终止请求执行剩余 cancel/flush，完成后发送 delayed-free 回执
3. 执行现有 input preparation，得到准确 num_computed_tokens
4. CPU planner 检查满块、冻结请求级计划并向 D2H stream 提交本轮 INFLIGHT
5. 使用本轮 stable_prefix 构造 metadata
6. replay Target/MTP graph
7. 在最后一个 KV writer 后 record source_ready
```

步骤 4 只在 D2H stream 排队，不等待本轮复制完成，因此可与步骤 6 重叠。步骤 1 是
正常热路径唯一等待上一轮满块 D2H 的位置；如果复制已经被上一轮模型图掩盖，它立即
返回，否则只等待剩余时间。没有 active INFLIGHT 时步骤 1 不执行 D2H event wait。

无 forward invocation 如果存在 INFLIGHT 或 finished/preempted cleanup，仍需执行相应
retire/flush；否则可以直接 no-op。它不启动普通 planner，也不记录 `source_ready`。
capture/dummy invocation 不属于真实 outer iteration。

## 安全不变量

1. 满块判断只读取经过现有流程更新后的 `num_computed_tokens`。
2. 普通 Decode 与 MTP 不维护两套长度或两套满块确认流程。
3. `stable_prefix` 只能在 D2H completion 和跨 TP 依赖之后增加，且每次增加 128。
4. INFLIGHT block 不能同时作为 Host-backed sparse history 使用。
5. D2H(A) 期间所有新 KV 只能写 B；A 再次成为写入页前必须完成 D2H。
6. copy stream 读取 A 前必须等待对应 `source_ready`。
7. generation 不匹配的 completion 不能推进新请求状态。
8. HBM source、Host destination 和 descriptor 在 INFLIGHT 完成前不能复用。
9. inactive、dummy 和越界描述符长度必须为零且地址安全。
10. 同一 logical block 每层 K/V、每个 generation 最多成功卸载一次。
11. rejected token 不能触发 D2H 或推进 `stable_prefix`。
12. Host→HBM restore 不能覆盖 resident tail 或 INFLIGHT A/B。
13. 所有 TP rank 必须冻结相同批次状态，并只对真实 INFLIGHT 以一致顺序参加 completion
    collective。
14. graph capture/dummy run 不能推进真实状态。
15. 请求释放或 slot 复用前必须完成 delayed-free/flush。
16. 没有 lifecycle provider 时不能启用 delayed Nano。
17. `num_computed_tokens - stable_prefix` 不得达到两个完整块。
18. CPU planner 只能读取生命周期维护的 CPU metadata，不能触发每轮 NPU→CPU 读回。
19. 请求级 plan 在 D2H stream 消费前不可覆盖；全层描述符在上一批 retire 前不可复用。
20. D2H 提交或执行失败时不能推进 `stable_prefix` 或复用相关资源，所有 TP rank 必须
    协同报错退出；第一版不重试或恢复。
21. 无 forward control tick 不能推进 source-ready epoch 或轮转 source-event 槽。

## 文件级适配清单

### `vllm_ascend/attention/sfa_kv_offload.py`

- metadata builder 使用 manager 提供的 `stable_prefix`。
- block table 和 `logical_lens` 支持 128--255 行 tail。
- Nano `exec_kv` 使用 `KvRmsNormRopeCache` 直接写 ring。
- 删除逐 token `offload_new_kv` 和逐层 full-block D2H。
- 删除稳态逐层 Host→HBM tail restore。
- tail residency mask 仅在共享 indexer owner 处更新一次。
- MTP 后续 step 保留真实 tail 可见长度。

### `sparse_kv_offload_manager.py`

- 保存 generation、stable-prefix 和单批 INFLIGHT 状态。
- 维护 planner 所需的 CPU generation/stable-prefix 视图。
- 预分配 pinned Host 请求级 plan、device plan 和全层 descriptor buffer。
- 增加图外 plan/launch、retire 和 terminal flush 接口。
- D2H stream 顺序执行 plan H2D、全层描述符展开、source-ready wait 和异步复制。
- pool capacity 使用完整可分配容量。
- 删除 PENDING、finalize、pending snapshot 和旧逐层 TP broadcast。

### `model_runner_v1.py`

- 在现有 `num_computed_tokens` 更新完成后调用 CPU planner。
- 不在 rejection 后调用 `finalize_nano_pending_blocks()`。
- graph 前 retire 上一批并异步 launch 新一批。
- graph/proposal 完成后记录 source-ready。
- 在 `_update_states()` 可能复用资源前捕获 terminal/preemption，先标记保护，再按
  retire → cancel/flush 顺序执行 cleanup。
- no-forward、finished 和 preempted 路径接入 cleanup。
- stable-prefix 更新排在跨 TP completion 之后。

### `sfa_pd_rd2h` connector scheduler/worker

- 接入 delayed-free，回执前不释放 block、slot、binding 或 generation。
- PD handoff 和 terminal Host 消费先 flush/等待。
- 无 lifecycle provider 或无 completion 依赖的运行中 Host consumer 明确拒绝。

### `llm_base_proposer.py`

- capture/runtime 使用相同 per-step metadata 布局。
- 整个 proposal 固定使用同一个 `stable_prefix`。
- step 1+ 使用实际 positions，TopK 复用保留真实 tail 长度。
- dummy run 不修改真实异步状态。

### `acl_graph.py`

- 不加入 Nano D2H fork/join。
- 保留其他 offloader 自身需要的 graph 生命周期逻辑。

## 测试计划

### 单元测试

- `127 → 128`：下一轮准确的 `num_computed_tokens=128` 时启动一个满块 D2H。
- sampler 产生第 128 个 token 但尚未 forward 时，`num_computed_tokens=127`，不启动。
- MTP `126 + 4` 计算、接受 2 行：下一轮 `num_computed_tokens=128`，启动 D2H。
- MTP 同样输入只接受 1 行：下一轮值为 127，不启动。
- ordinary Decode/MTP 共用同一满块判断入口。
- 检测 `num_computed_tokens >= stable_prefix + 256` 时失败，不静默漏块。
- 多个请求同轮填满时合并为一批固定形状描述符。
- R1/R2 连续两轮填满：先退休 R1，再复用描述符启动 R2。
- 第 N+1 轮 `stable_prefix` 不推进，Attention 包含 A+B。
- 第 N+2 轮 completion 后前移 128，A 进入 history，dense tail 只包含 B。
- `stable_prefix=128 < configured_topk` 时，A 通过 dense-identity history 参与 Attention，
  不会因 `attention_cache_tokens=0` 被遗漏。
- plan H2D 和描述符展开必须先于 D2H；copy stream 在 source-ready 前不能读取 A。
- generation 变化使旧 completion 失效。
- D2H failure 不推进 CPU/device `stable_prefix`、不释放 source/destination，并触发所有
  TP rank 协同报错退出。
- TP>1 各 rank 冻结相同的 batch epoch、generation 和 target-prefix；仅真实 INFLIGHT
  执行 completion collective，空轮次不执行。
- 最大两个合法 graph-padding slot 不被 `max_num_reqs` 错误过滤。
- dummy、inactive 和 rejected rows 不产生非零复制长度。
- MTP step 1+ 和 TopK 复用保持真实 tail 可见。
- Host restore 不覆盖 resident/INFLIGHT tail。
- 请求结束时按 Host 消费需要执行取消或一次性 flush。
- INFLIGHT 请求结束/抢占后，`finished_sending` 前资源不复用。
- 暂时未调度的请求继续保留 Nano slot。
- 暂时未调度后重新出现的满块使用最新 compute frontier，仍能安全启动 D2H。
- 无 forward invocation 可以完成已有 INFLIGHT 和 terminal cleanup。
- 无 forward invocation 不记录 source-ready，也不改变下一次真实 compute 使用的 event 槽。

### 图模式测试

- graph 内不存在 Nano full-block `OffloadSparseCopy`、`join_nano_d2h` 和逐层 restore。
- graph 内用融合 `KvRmsNormRopeCache` 替换分离 KV 写入，不新增 Attention 或 tail copy。
- capture/dummy run 不推进真实状态。
- Target、MTP step 0 和后续 step 使用正确 metadata。
- 不同 batch shape、padding 和请求重排不污染 generation。
- MTP 多个 draft step 每个 outer iteration 只 launch/retire 一次。

### NPU 精度测试

- 序列长度覆盖 126、127、128、129、255、256。
- 普通 Decode 与 MTP 覆盖全接受、部分拒绝和全部拒绝。
- mixed short/long request batch。
- ACL graph 与 eager 结果一致。
- 人为延迟 D2H 时，第 N+1 轮仍从 A+B 得到正确结果。

### Profiling 验收

- full-block D2H 位于图外 copy stream，不出现在每层主计算流。
- D2H(A) 与第 N+1 轮模型图重叠。
- 第 N+2 轮只在 D2H 未被模型图掩盖时等待剩余时间。
- 每个满块、每层 K/V 的非零 D2H 只发生一次。
- 每层不存在 tail restore `ScatterElements`；请求级 residency 更新只出现一次。
- CPU planner 不产生逐请求 NPU→CPU 同步，不新增 NPU planner kernel。
- 分别记录 accepted-count event wait、CPU planner、请求级 plan H2D 和 device descriptor
  展开的实际开销。
- TP>1 每批真实 INFLIGHT 恰好执行一次 completion collective；无 INFLIGHT 时不执行。

## 完成标准

- 普通 Decode、MTP eager 和 ACL graph 精度通过。
- 只使用现有 `num_computed_tokens` 判断满块，没有新增平行长度或 rejection finalize。
- 第 N+1 轮确实计算两个 tail page，且复制源不被覆盖。
- `stable_prefix` 只在 D2H 完成后推进。
- 所有 TP rank 只在 TP0 completion 之后推进相同状态。
- TP>1 无 INFLIGHT 的轮次不增加 completion collective。
- 请求结束、preemption、PD handoff 和资源复用没有未完成 DMA。
- profiling 证明 D2H 离开图内关键路径，且未引入热路径 NPU→CPU 同步。
- profiling 证明请求级 planner 没有按层展开，也没有新增 Python 后台线程。
- 不保留逐层 tail restore、逐层 TP completion broadcast 或每层请求级
  `ScatterElements`。
