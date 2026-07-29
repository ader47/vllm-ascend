# DSA 稀疏卸载测试入口

本目录是 v0.23 DSA 的 tester-facing 验收入口。脚本默认以 GLM-5.1 为首要
模型，DeepSeek-V3.2 使用相同配置做强制回归。

| 文件 | 用途 |
|---|---|
| `simple_prompt_test.py` | 基线隔离、cache-init、eager 和 graph 冒烟 |
| `qa_dataset_test.py` | LongBench 风格数据集精度回归，可切 disabled/eager/graph |
| `eval_dataset_acc_score.py` | 对 QA 脚本生成的 JSONL 计算 LongBench 指标 |

当前架构和稳定 ABI 见
[DSA 稀疏卸载详细设计](../../docs/source/developer_guide/Design_Documents/dsa_offload_design.md)。

## 1. 前置条件

- 配套 vLLM v0.23 和当前 vLLM-Ascend 已安装；
- 当前 checkout 的自定义算子已经完整编译并重新安装；
- Ascend 910C 首版使用 PP=1、DCP=1、PCP=1；
- `block_size=128`；
- 关闭 async scheduling、chunked prefill、prefix cache、MTP/speculative
  decoding、KV transfer 和 KV-cache metrics/events；
- 正式精度和性能测试关闭 DSA trace points。

脚本只设置必要的 vLLM/vLLM-Ascend 原生环境变量。DSA 功能参数全部位于
`additional_config["dsa_sparse_config"]`，不再通过零散 DSA 环境变量控制。

首次拉起应看到且只看到一份：

```text
================ DSA HBM CACHE CAPACITY REPORT ================
```

报告中的 Indexer blocks 应等于 resident MLA base blocks 乘
`indexer_mla_block_ratio`。

## 2. 快速冒烟

`simple_prompt_test.py` 延续 v0.19 的直接用法，不提供一长串 CLI 参数。
先修改文件顶部“用户配置”区：

```python
MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a8"
RUN_MODE = "eager"  # disabled / cache-init / eager / graph
PROMPTS = [
    "第一个测试文本",
    "第二个测试文本",
]
MAX_NUM_SEQS = 2
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 8192
RESULT_JSON = None  # 需要保存对照 token IDs 时填写路径
```

然后直接运行：

```bash
python examples/dsa_demo/simple_prompt_test.py
```

四种模式的作用：

- `disabled`：不传 `dsa_sparse_config`，验证 DSA 修改未影响原生路径；
- `cache-init`：只构造 `LLM`，验证 Indexer/resident MLA 双平面初始化；
- `eager`：执行 DSA LIDU/KSC/SFA-Offload 和满块 dump；
- `graph`：执行 `VLLM_COMPILE + FULL_DECODE_ONLY`。

`cache-init` 不执行 `generate()`，因此不能证明数据面算子已经运行。结果 JSON
也不捕获 engine-core 子进程日志；容量报告是否出现且只出现一次，必须在
完整控制台日志中单独核对。

默认短 prompt 主要覆盖 DENSE。构造 sparse/ENTER 或混合预算 batch 时，直接
替换 `PROMPTS` 并相应增大 `MAX_NUM_SEQS`、`MAX_MODEL_LEN` 和
`MAX_NUM_BATCHED_TOKENS`。实际阈值覆盖应使用 tokenizer token 长度，不要把
字符数当 token 数。

graph 模式下，单 token DENSE/ENTER/SPARSE 任意混排可进入统一 FULL graph；
prefill、multi-token、capture-size miss 或其他原生动态 blocker 会被 DSA
显式送入 true eager，不会执行一张仅覆盖部分 DSA metadata 的 piecewise
graph。非法 async 配置由 `tests/ut/dsa_offload/test_config.py` 覆盖，不再
为了这一项给最小脚本增加独立运行模式。

## 3. QA 数据集回归

`qa_dataset_test.py` 延续旧项目对测试同事更直接的用法：修改文件顶部“用户
配置”区，不强制记忆大量 CLI 参数。至少设置：

```python
MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a8"
DATASET_FILE = "/home/data/longbench/multifieldqa_zh.jsonl"
DATASET_START = 0
DATASET_LIMIT = 100
RESULT_DIR = "LongBenchResult/glm51_dsa"
RUN_MODE = "eager"
BATCH_SIZE = 4
MAX_MODEL_LEN = 131072
MAX_NUM_BATCHED_TOKENS = 131072
```

然后运行：

```bash
python examples/dsa_demo/qa_dataset_test.py
```

建议对同一数据切片跑三次：

1. `RUN_MODE="disabled"`：原生基线；
2. `RUN_MODE="eager"`：DSA eager；
3. `RUN_MODE="graph"`：DSA FULL decode graph。

脚本自动在 `RESULT_DIR` 下增加 `disabled/eager/graph` 子目录，不会让三次
结果相互覆盖。

输出 JSONL 保留：

- `pred`；
- `answers`；
- `all_classes`；
- LongBench `length`；
- `sample_id`、`sample_index`；
- 实际 `prompt_tokens`。

这既可用于自动评分，也可直接定位异常请求。若一批 prompt token 总数超过
`max_num_batched_tokens`，脚本会提示 scheduler 可能分多轮完成 prefill；
当前 chunked prefill 仍保持关闭。

脚本默认 `MIN_TOKENS=0`，不会强制模型绕过正常 EOS。评分器也仅按
LongBench 官方规则对 `trec/triviaqa/samsum/lsht` 截取首行；QA 和
retrieval 任务保留完整输出参与评分，以免掩盖后续乱码、重复 prompt 或异常
提前结束。

## 4. LongBench 评分

评分脚本需要环境中存在 `jieba`、`fuzzywuzzy` 和 `rouge`。对一个目录：

```bash
python examples/dsa_demo/eval_dataset_acc_score.py \
  --result-path \
  examples/dsa_demo/LongBenchResult/glm51_dsa/eager
```

对单个文件：

```bash
python examples/dsa_demo/eval_dataset_acc_score.py \
  --result-path /path/to/multifieldqa_zh.jsonl
```

增加 `--longbench-e` 可输出 `0-4k`、`4-8k` 和 `8k+` 三档分数。

## 5. 最小回归矩阵

| 维度 | 至少覆盖 |
|---|---|
| 模式 | disabled、eager、graph |
| batch | bsz=1、bsz>1 |
| 长度 | 超短、DENSE、三档 resident budget |
| row mode | 全 DENSE、全 SPARSE、DENSE/ENTER/SPARSE mixed |
| graph | active=captured、active<captured 的 PAD |
| dump | prefill 多满块、decode 跨满块 |
| 生命周期 | 请求结束、InputBatch condense/reorder、stable row 复用 |
| 模型 | GLM-5.1 主验收、DeepSeek-V3.2 回归 |

不要只看“进程跑完”。至少核对：

- 无乱码和异常提前 EOS；
- 同一请求在 bsz=1 与 bsz>1 下没有明显语义断裂；
- eager 与 graph 没有单行串扰；
- finish reason 与输出 token 数合理；
- 容量报告的 blocks、tokens、bytes 自洽；
- 没有 silent fallback 到未启用 DSA 的路径。

## 6. 问题反馈工件

出现问题时保留：

- vLLM 和 vLLM-Ascend commit；
- 完整脚本配置或结果 JSON；
- 模型 architecture、量化方式、TP/EP/DP；
- prompt token 长度和出错 sample index；
- 从 `non-default args` 到异常结束的完整日志；
- DSA HBM 容量报告；
- eager/graph 对照 token IDs 与 finish reason；
- CANN、torch、torch-npu、设备型号；
- 性能问题对应的 profiler 目录和测试场景。

`trace_points` 当前是拉起期可解析的预留合同，尚无稳定日志 consumer。不要
把“没有 trace 输出”解释成 DSA 未运行；确认路径应结合容量报告、自定义
算子 profiling 和输出结果。

## 7. 当前不支持

- chunked prefill 与 prefill/decode mixed batch；
- async scheduling；
- prefix cache；
- preemption/resume；
- speculative decoding/MTP；
- KV transfer connector；
- context parallel 和 pipeline parallel；
- KV-cache metrics/events；
- A5 设备算子验收。

这些能力会在后续按独立合同逐项扩展。在相应实现和回归完成前，不要通过
删除启动期校验强行打开。
