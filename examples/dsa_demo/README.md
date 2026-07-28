# DSA 稀疏卸载 v0.23 迁移 Demo

本目录当前用于验证迁移阶段 P2 的 KV-cache 控制面，不用于证明 DSA
稀疏卸载数据面已经接通。

P2 已把 Indexer dense plane 与 MLA resident plane 拆成独立 spec、group、
tensor 和物理 block pool。P4-P5 尚未接通请求阶段与算子 consumer，因此
开启 DSA 后当前只允许做 cache 初始化验收，不运行 `generate`。

## 1. 前置条件

- 已安装匹配的 vLLM v0.23 和当前 vLLM-Ascend checkout；
- 当前首要验证模型为 GLM-5.1；
- DeepSeek-V3.2 是强制回归模型；
- 使用 Ascend 910C；
- 首版使用 PP=1、DCP=1、PCP=1，并关闭 KV-cache metrics/events；
- 模型支持的量化方式通过 `--quantization` 指定，默认是 `ascend`。

如果使用 editable 安装，本阶段只有 Python 文件变化，通常无需重新编译
自定义算子；普通 wheel 安装需要重新安装 vLLM-Ascend。

## 2. 三个 P2 用例

下面以 GLM-5.1 模型为例。输出分别保存为 JSON，便于核对启动结果和 token
序列。

### 2.1 DSA 关闭：原生基线非回归

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode disabled \
  --result-json /tmp/dsa-p2-disabled.json
```

预期：正常加载并生成。

### 2.2 DSA 开启：双平面 cache 初始化

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode cache-init \
  --result-json /tmp/dsa-p2-cache-init.json
```

预期：类型化配置与模型能力校验通过，日志恰好出现一份
`DSA HBM CACHE CAPACITY REPORT`，其中包含两个物理平面：

- `Indexer dense plane` 的 block 数为 `MLA resident plane` 的
  `indexer_mla_block_ratio` 倍；
- `KVCacheConfig.num_blocks` 等于 MLA resident plane 的 block 数；
- resident MLA 与 Indexer cache 均成功绑定到各自的
  `static_forward_context` 模块，不出现同层多 cache 的
  `NotImplementedError`；
- 脚本明确打印 `generation was intentionally skipped`。

该结果只说明 P2 cache 控制面工作正常，不说明已经发生 DRAM dump 或
LIDU/KSC/SFA-Offload 计算。

### 2.3 不支持组合：拒绝 async scheduling

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode reject-async \
  --result-json /tmp/dsa-p2-reject-async.json
```

预期：在加载模型权重前失败，脚本识别到以下错误后以成功状态退出：

```text
DSA sparse offload currently requires async_scheduling=False
```

## 3. 结果核对

`disabled` 和 `cache-init` 都成功后，核对：

- 两次使用相同模型、prompt、seed 和采样参数；
- `disabled` 的输出结构、首 token 和 finish reason 是否合理；
- `cache-init` 日志中是否恰好包含一份 DSA 容量报告；
- Indexer/MLA block 数、token 容量和总分配字节是否自洽；
- `reject-async` 是否准确命中预期错误，而不是在更早位置异常退出。

GLM-5.1 W4A8 在 EP/TP 多卡环境下，即使 `temperature=0` 和 seed 相同，
两个独立进程的原生 `disabled` 结果也可能从 decode 阶段开始分叉。因此，
只有先证明同一 `disabled` 模式可重复，才能把 disabled/enabled 的完整
token IDs 不一致视为配置副作用；不能直接用跨进程逐 token 相等作为 P2
验收门槛。

请保留以下信息用于问题定位：

- 从 `non-default args` 到 KV cache 初始化完成的日志；
- 三份结果 JSON 和完整 DSA 容量报告；
- vLLM、vLLM-Ascend commit；
- 模型路径与 architecture；
- CANN、torch、torch-npu 版本。

## 4. 暂未迁移的旧 demo

v0.19 的 `qa_dataset_test.py` 和 `eval_dataset_acc_score.py` 用于完整 DSA
数据面的精度回归。它们将在 P5 接通 Indexer/MLA 解耦、DRAM dump 和
LIDU/KSC/SFA-Offload 后迁移；现在运行这类测试只能测到 v0.23 原生 packed
数据面尚未接通，容易产生错误结论。
