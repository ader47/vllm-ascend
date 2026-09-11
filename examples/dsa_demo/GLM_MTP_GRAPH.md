# GLM：target graph + 三轮 MTP 合并 graph

本次改动仅涉及 Python，不改变 LIM ABI、C8 布局、target 的 78 层图或
MTP 的 full BF16 KV cache。基于已编译好的当前分支使用，无需为本次改动重新编译算子。
这是待 Ascend 真机验收的首版，CPU 测试不能证明 CANN capture/replay 或整网精度已经通过。

运行前需要同步框架文件 `vllm_ascend/spec_decode/llm_base_proposer.py`、
`vllm_ascend/spec_decode/glm_mtp_graph.py`，以及
`examples/dsa_demo/simple_prompt_test_dp16.py`、
`examples/dsa_demo/simple_prompt_test_tp8_mtp_graph.py`；只复制新测试入口不够。

## 开关与边界

`simple_prompt_test_dp16.py` 的配置区：

```python
MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
RUN_MODE = "graph"
TENSOR_PARALLEL_SIZE = 1
DATA_PARALLEL_SIZE = 16
ENABLE_MTP = True
MTP_NUM_SPECULATIVE_TOKENS = 3
ENABLE_MTP_GRAPH = True
ENABLE_CHUNKED_PREFILL = False
ENABLE_PROFILE = False
```

其它 prompt、长度上限、token budget、显存比例等沿用你已跑通的配置。
首次建议每 rank BS1、短 prompt、输出 64 token；不要恢复到已知有问题的 4096 长度配置。

实际入口是 `speculative_config.enforce_eager=False`。改回
`ENABLE_MTP_GRAPH=False` 即得到 **target graph + drafter eager** 对照组。
`RUN_MODE="eager"` 时两者仍为 eager；仅打开 `ENABLE_MTP_GRAPH` 不会打开 MTP。

基线与 offload 对照统一保留 MTP 的 BF16 全量 MLA/Indexer cache：即使不传
`dsa_sparse_config`，MTP 层也会关闭 SFA C8 和 LI C8，target 层仍遵循原来的
C8 开关。这样避免 BF16 MTP 投影将 packed FP8 cache 传给非量化 MLA Prolog。
此策略与 MTP graph 开关独立；同步 `vllm_ascend/attention/sfa_v1.py` 和
`vllm_ascend/worker/model_runner_v1.py` 后重启进程即可生效，无需重新编译算子。

限定 Ascend A5、`glm_moe_dsa`、DSA offload、MTP3，不再按 TP/DP 数值设置白名单。
target/draft TP 必须相同；拓扑仍需满足模型、显存和通信后端本身的约束。
允许配置不代表所有拓扑均已通过真机验证，特别是混合 TP+DP 的 MC2/EP 路径。
PP/PCP/DCP 均为 1、同步调度、非 SP/共享专家 DP/FlashComm2、非动态 EPLB/ENPU。
不支持额外的 fine-grained lm-head TP 功能；常规 TP 词表分片和 logits all-gather 保留。
target 配置为 `FULL_DECODE_ONLY`。其它 GLM 配置继续 drafter eager，并打印原因。
只对每请求 4 个 verification token 的纯 decode 批次选图；prefill、混合/非均匀批次
或任一 DP rank 不满足条件时，全 DP 组回退 eager。

每个 capture size 分别有一张 target 图和一张 MTP 图，并非整个进程总共只创建两张图。
MTP 图包含 padding 清理、后两轮 position/seq_len/slot 更新、三轮 RoPE gather、
可选 C8 grouping metadata 更新，以及三轮 forward、logits/greedy 采样和下一轮输入更新。
metadata 按 capture size 缓存，query layout 和设备输出地址固定；CPU seq_len 原地更新。
DP metadata 同步、首轮输入整理及 staging copy、CPU bookkeeping 和 target rejection sampling 仍在图外。
RoPE 使用 `index_select` 直接写入固定 cos/sin 缓冲区，target 的公共 RoPE 查表也复用这项优化。

### 前处理优化验收

不需要重新编译算子；同步 Python 源码后重新启动进程，重新捕获图。
继续用 DP2TP4、24k、每 DP BS1、chunked prefill 关闭的同一组输入测试：

1. 先关闭 profile，比较旧版 MTP graph、新版 MTP graph 和 MTP eager 的 token_ids。
2. 检查日志 `Replaying separate merged 3-step draft graph (with device preprocessing)`。
3. 再采集稳态 profile：propose 路径图外不应每步重建三套 builder/RoPE；
   RoPE gather、后两轮 slot 更新应出现在同一张 MTP 图开头。
4. 使用同一完整 step 边界比较 ITL，不把 propose 变短或 take_draft 等待变长单独视为收益。
5. 补测不同接受位置、batch 缩小、空闲 DP、切换 graph size 和 eager/full 切换。

Python 修改涉及 `model_runner_v1.py`、`llm_base_proposer.py`、`glm_mtp_graph.py`、
`ops/rotary_embedding.py`，应一起同步。LIM/QSFA/scatter 算子及 ABI 未改。
CPU 测试不能证明 Ascend 上的 capture/replay 正确性或实际加速。

## 通用 TP/DP 验收入口

统一使用 `simple_prompt_test_dp16.py`（名字保留兼容，启动逻辑不限 DP16）。
例如单机 8 卡 DP4TP2，修改顶部配置：

```python
RUN_MODE = "graph"
TENSOR_PARALLEL_SIZE = 2
DATA_PARALLEL_SIZE = 4
ENABLE_MTP = True
ENABLE_MTP_GRAPH = True
MTP_NUM_SPECULATIVE_TOKENS = 3
ENABLE_CHUNKED_PREFILL = False
ENABLE_PROFILE = False
MAX_NUM_SEQS = 1
MAX_MODEL_LEN = 25600
MAX_NUM_BATCHED_TOKENS = 25600
MAX_TOKENS = 64
PROMPTS = ["请用一句话介绍你自己。"] * DATA_PARALLEL_SIZE
RESULT_JSON = "mtp_graph.json"
```

确认模型路径及 `ENABLE_EXPERT_PARALLEL=True`、`ENABLE_A5_PACKED_C8_DSA=True`。
运行 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/dsa_demo/simple_prompt_test_dp16.py`。
脚本自动匹配 draft TP，并关闭 SP/共享专家 DP/FlashComm2，eager 对照使用同一布局。
仅将 `ENABLE_MTP_GRAPH=False`、`RESULT_JSON="mtp_eager.json"` 后运行对照。
结果按 DP rank 分文件保存。比较 token_ids，并确认 merged draft graph replay 日志。
之后用不同长度 prompt/输出长度验证部分 DP 提前结束，再扩大并发和上下文长度。
其它拓扑只改 TP/DP；跨机仍使用下文节点启动参数，每个 TP 组需完整落在一台机器上。
这不是对跨机 TP、PP、PCP/DCP、异步调度或 SP 的新增支持。

## 单机 TP8DP1 先行验收

新增 `simple_prompt_test_tp8_mtp_graph.py`，复用 dp16 脚本的启动逻辑，
不修改原有单机和双机脚本的用户配置。默认模型路径为
`/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4`，target/drafter 均 TP8、DP1EP8，
BS1、短 prompt、输出上限 64 token、max_model_len 与 token budget 均为 25600，
关闭 chunked prefill、profile、SP（含 compiler pass）、FlashComm2 和 reduce-sample。

在一台 8 张卡空闲的 Ascend950DT 容器内，从对应 editable 仓库根目录顺序运行：

```bash
unset ASCEND_LAUNCH_BLOCKING
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python examples/dsa_demo/simple_prompt_test_tp8_mtp_graph.py --mtp-eager \
  2>&1 | tee tp8_mtp_eager.log

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python examples/dsa_demo/simple_prompt_test_tp8_mtp_graph.py \
  2>&1 | tee tp8_mtp_graph.log
```

不需要两机 IP、node-rank 或跨机端口参数；通信网卡仍使用本机有效配置。
两次运行分别保存 `tp8_mtp_eager.json` 和 `tp8_mtp_graph.json`，然后比较：

```bash
python -c 'import json; a=json.load(open("tp8_mtp_eager.json")); b=json.load(open("tp8_mtp_graph.json")); assert [(r["token_ids"], r["finish_reason"]) for r in a["outputs"]] == [(r["token_ids"], r["finish_reason"]) for r in b["outputs"]], "FAIL: eager/graph mismatch"; print("PASS: eager/graph token IDs and finish reasons match")'
grep -F '[GLM MTP graph]' tp8_mtp_graph.log
```

图组应出现 `Enabled ... target_tp=8 draft_tp=8 dp=1`、`Captured 3-step draft graph`
以及 `Replaying separate merged 3-step draft graph`。只有 target 的通用 replay 日志不算通过。
BS1 的 4 个 verification token 不必人为改成 8 个：MC2 mask 单独补齐到 TP8，
三轮有效通信行数为 4 → 1 → 1，补齐行始终无效。

短 prompt 通过后，可在新脚本中将 `smoke.PROMPTS` 改成 `smoke.chinese_20k[:1]`；
再增加并发，需同时增加 prompt 数量和 `smoke.MAX_NUM_SEQS`。结果文件名固定，重复运行前
自行保留需要比较的文件。单机通过不能替代 DP16 的跨机 EP 与空闲 rank 验收。

## 双机启动

两边同步本次 Python 改动并保留一致的测试配置；从已安装为 editable 的对应仓库根目录启动。
端口需空闲、两端一致。以下网卡来自之前的连通性结果，若环境更换需相应调整。

`.13` 容器：

```bash
unset ASCEND_LAUNCH_BLOCKING
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_HOST_IP=141.61.33.13 HCCL_IF_IP=141.61.33.13 \
HCCL_SOCKET_IFNAME=enp34s0f1 GLOO_SOCKET_IFNAME=enp34s0f1 TP_SOCKET_IFNAME=enp34s0f1 \
python examples/dsa_demo/simple_prompt_test_dp16.py \
  --node-size 2 --node-rank 0 \
  --master-addr 141.61.33.13 --master-port 29500 --sync-port 29600 \
  2>&1 | tee dp16_mtp_graph_13.log
```

`.21` 容器：

```bash
unset ASCEND_LAUNCH_BLOCKING
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_HOST_IP=141.61.33.21 HCCL_IF_IP=141.61.33.21 \
HCCL_SOCKET_IFNAME=eth4 GLOO_SOCKET_IFNAME=eth4 TP_SOCKET_IFNAME=eth4 \
python examples/dsa_demo/simple_prompt_test_dp16.py \
  --node-size 2 --node-rank 1 \
  --master-addr 141.61.33.13 --master-port 29500 --sync-port 29600 \
  2>&1 | tee dp16_mtp_graph_21.log
```

## 验收

1. 两机都出现 `[GLM MTP graph] Captured 3-step draft graph`，随后出现
   `[GLM MTP graph] Replaying separate merged 3-step draft graph`。
   仅有通用 `Replaying aclgraph` 不能说明 MTP 已入图。
2. 同一输入和采样配置，分别 `ENABLE_MTP_GRAPH=False/True`，通过 `RESULT_JSON`
   保存各 rank 的 token IDs 并逐 token 比较。例如 `./mtp_eager.json` 会生成
   `mtp_eager.dp0.json` 等文件；图组用另一个文件名前缀。
3. BS1 短 prompt 通过后再测 20k、70k、每 rank BS8。增加不同长度/不同输出长度的请求，
   覆盖 rank 提前结束、请求数下降、block 边界、MTP 不同接受长度。
4. 最后开 profile，确认三个 draft iteration 位于同一次 MTP graph replay 中，
   比较完整 step 时间、MTP 时间、接受长度与吞吐。不能将 96 ms − 58 ms 全部当成可消除的开销。

若 capture 或 replay 失败，先保留两机第一条异常和完整 traceback，关闭
`ENABLE_MTP_GRAPH` 恢复基线；不要删除算子断言或修改 padding 的 `-1` slot。

## 本地检查范围

CPU 测试覆盖固定地址 metadata、多种 TP/BS 的 MC2 mask 补齐、两路 hidden states 保持独立、
批次缩小/空闲、DP 选图一致性以及单机脚本配置。另有真实 8 进程 Gloo 测试，
覆盖 TP1DP8、TP2DP4、TP4DP2、TP8DP1 的子通信组、不同 DP 的批次缩小/空闲，
执行三轮 TP all-reduce 和 logits all-gather（包括 reduce-sample 分支），
各 rank 的输出均与完整词表 eager 参考结果比较。
这些测试不执行 CANN/HCCL/MC2 算子，不能代替上述 Ascend 真机验收。
