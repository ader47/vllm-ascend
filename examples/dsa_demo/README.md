# DSA 稀疏卸载测试入口

本目录是 v0.23 DSA 的 tester-facing 验收入口。脚本默认以 GLM-5.1 为首要
模型，DeepSeek-V3.2 使用相同配置做强制回归。

| 文件 | 用途 |
|---|---|
| `simple_prompt_test.py` | 基线隔离、cache-init、eager 和 graph 冒烟 |
| `qa_dataset_test.py` | LongBench 风格数据集精度回归，可切 disabled/eager/graph |
| `eval_dataset_acc_score.py` | 对 QA 脚本生成的 JSONL 计算 LongBench 指标 |
| `serve_dsa.sh` | 以 OpenAI 兼容在线服务拉起 DSA eager/graph |
| `stream_chat_client.py` | 交互式多轮流式客户端 |
| `online_long_context_dataset_test.py` | 在线长上下文数据集分桶回归 |

当前架构和稳定 ABI 见
[DSA 稀疏卸载详细设计](../../docs/source/developer_guide/Design_Documents/dsa_offload_design.md)。

## 1. 前置条件

- 配套 vLLM v0.23 和当前 vLLM-Ascend 已安装；
- 当前 checkout 的自定义算子已经完整编译并重新安装；
- Ascend 910C 首版使用 PP=1、DCP=1、PCP=1；
- `block_size=128`；
- 关闭 async scheduling、prefix cache、MTP/speculative decoding、KV
  transfer 和 KV-cache metrics/events；
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
ENABLE_CHUNKED_PREFILL = False
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

chunked prefill 复用 vLLM v0.23 原生调度。把
`ENABLE_CHUNKED_PREFILL=True`，并把 `MAX_NUM_BATCHED_TOKENS` 调到小于
单条长 prompt 的 token 数，即可覆盖多个 chunk。DSA 仍保持
prefill/decode phase barrier：每个 prefill chunk 走 eager，最后一个 chunk
完成并同步 dump 后才释放 resident 满块，后续单 token decode 可正常进入
FULL graph。`scheduler_reserve_full_isl` 必须保持默认的 `True`，保证首个
chunk 入场前完整 prompt 能同时容纳于 Indexer 与 resident MLA 两个 dense
plane。

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
ENABLE_CHUNKED_PREFILL = False
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
打开 `ENABLE_CHUNKED_PREFILL` 后，单个长 prompt 也可以按 token budget
切成多个 prefill chunk。

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
| chunked prefill | 中间 chunk、最终 chunk、完成后首个 decode |
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

## 7. 在线流式服务

先修改 `serve_dsa.sh` 顶部用户配置。首次在线验证建议从
`RUN_MODE="eager"` 开始，确认请求生命周期、连续请求和流式返回正常后，再
切换为 `RUN_MODE="graph"`：

```bash
bash examples/dsa_demo/serve_dsa.sh 2>&1 | tee dsa-online.log
```

服务就绪后，另开终端检查：

```bash
curl -H "Authorization: Bearer EMPTY" \
  http://127.0.0.1:8000/v1/models
```

运行交互式流式客户端：

```bash
python examples/dsa_demo/stream_chat_client.py
```

客户端会保留并在下一轮重新发送完整消息历史。设置
`LONG_CONTEXT_FILE="/path/to/long_context.txt"` 可以把长文本加入 system
消息；只有实际 token 长度超过 `sparse_activation_tokens`，才能验证
ENTER/SPARSE 路径，而不是只验证在线 API 和 DENSE 路径。由于 DSA 当前关闭
prefix cache，多轮长上下文测试每次都会重新 prefill 完整历史，这是当前
合同内的预期行为。

若要复现离线 `llm.generate()` 的原始小说续写，而不是让 chat template
把小说包装成一轮对话，可在 `stream_chat_client.py` 的用户配置区设置：

```python
from novel_dataset import chinese_40k

CONTINUATION_PROMPT = chinese_40k[0]
MAX_TOKENS = 512
```

此时脚本会调用流式 `/v1/completions`，将 `chinese_40k[0]` 原样作为
prompt，输出一次续写后退出。`chinese_40k` 在测试数据中是只包含一条
文本的 list，因此这里需要取 `[0]`。

不安装 Python OpenAI SDK 时，也可以直接观察 SSE 流：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1-dsa",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": true
  }'
```

在线冒烟至少核对：

- 启动日志中 DSA HBM 容量报告只出现一次；
- `/v1/models` 返回 `glm-5.1-dsa`；
- SSE/客户端内容逐步输出，而非等待完整回答后一次性打印；
- 同一短请求在在线 eager 与离线 eager 下无明显语义差异；
- 长上下文请求能够进入 DSA 算子路径；
- 连续多轮、`/clear` 后新请求以及请求结束后的 row 复用均无串扰；
- eager 通过后再复测 graph，比较 token IDs、结束原因和服务日志。

### 在线长上下文数据集

`online_long_context_dataset_test.py` 可直接复用已经拉起的在线服务。测试数据包
包含 LV-Eval 中文 multifield QA 的 16K/32K/64K/128K 档，以及 CLongEval
小说问答和键值检索的 small/medium/large 档。

先把数据包解压到服务器，例如：

```bash
unzip -q dsa_long_context_dataset_pack.zip -d /home/data
```

确认脚本顶部配置：

```python
API_BASE = "http://127.0.0.1:8000/v1"
MODEL_NAME = "glm-5.1-dsa"
LOCAL_TOKENIZER_PATH = None
DATASET_ROOT = "/home/data/dsa_long_context_dataset_pack"
RUN_LABEL = "dsa-eager-online"
ENABLE_THINKING = False
MAX_SAMPLES_PER_FILE = None
```

服务无需重启，直接运行：

```bash
python examples/dsa_demo/online_long_context_dataset_test.py
```

脚本按照 `dataset_specs()` 中的数据集顺序和 JSONL 原始行序完整处理样本，
不按长度选样或重排。`MAX_SAMPLES_PER_FILE = None` 表示完整处理每个文件；
仅在快速调试脚本时才设置正整数上限。脚本从 `/v1/models` 返回的模型
`root` 本地加载同一 tokenizer（不加载模型权重），提前计算 prompt token 数
并拦截超过 `max_model_len` 的样本。服务使用单独的 `--tokenizer` 路径时，
需要通过 `LOCAL_TOKENIZER_PATH` 显式指定。

第一条有效样本会额外调用一次服务端 `/tokenize`，确认本地 tokenizer、chat
template 和 `enable_thinking` 得到的计数完全一致；其余样本均在本地预检。
正式响应的 `usage.prompt_tokens` 会再次与预检值核对。样本按实际长度标记为
`0-6K`、`6-16K`、`16-32K`、`32-48K`、`48-64K`、`64-96K` 或
`96K+`。

每 `REQUEST_CONCURRENCY` 条样本组成一个客户端并发组。脚本会在发送前逐行
打印数据集、子集、样本 ID、任务要求、问题（或检索键）和 golden answer；
同时打印预检 token 数和长度区间。组完成后再打印正式响应中的实际 token
数。脚本不会人为安排 mixed-length 组合；实际进入同一个引擎 batch 的请求仍
由在线调度器决定。结果保存为 JSONL，并生成按数据集、任务类型和实际 token
档汇总的 `summary.json`。

当前接入的任务包括：

- CLongEval `long_story_qa`：长篇小说问答；
- CLongEval `key_passage_retrieval`：从 JSON 长上下文中检索指定键的值；
- LV-Eval `multifieldqa_zh_mixup`：多文章混合场景问答；
- 可选 LongBench `multifieldqa_zh`：长上下文问答。

数据包中的 NeedleBench 文件目前只是后续扩展所需的原始语料和 needle 定义，
尚未被该脚本组装成测试样本。

GLM-5.1 的 chat template 默认开启 thinking。短答案精度回归保持
`ENABLE_THINKING = False`；否则较小的 `MAX_TOKENS` 可能全部成为
reasoning token，最终答案为空。

`character_f1` 和 `contains_answer` 是不依赖额外包的冒烟指标，不等价于
LV-Eval 或 CLongEval 官方排行榜分数。正式精度对比应保持相同数据包、服务
参数和 `RUN_LABEL`，分别拉起 baseline、DSA eager 与 DSA graph 后运行。

## 8. 当前不支持

- prefill/decode mixed batch（chunked prefill 本身已支持）；
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
