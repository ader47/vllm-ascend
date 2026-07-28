# DSA 稀疏卸载 v0.23 迁移 Demo

本目录当前用于验证迁移阶段 P1 的配置控制面，不用于证明 DSA 稀疏卸载
数据面已经接通。

P2-P5 完成前，即使 `dsa_sparse_config.enabled=True`，框架仍使用 v0.23
原生 packed Indexer/MLA cache。此时没有 DSA 容量报告、DRAM dump 或
LIDU/KSC/SFA-Offload 调用是正常现象。

## 1. 前置条件

- 已安装匹配的 vLLM v0.23 和当前 vLLM-Ascend checkout；
- 当前首要验证模型为 GLM-5.1；
- DeepSeek-V3.2 是强制回归模型；
- 使用 Ascend 910C；
- 模型支持的量化方式通过 `--quantization` 指定，默认是 `ascend`。

如果使用 editable 安装，本阶段只有 Python 文件变化，通常无需重新编译
自定义算子；普通 wheel 安装需要重新安装 vLLM-Ascend。

## 2. 三个 P1 用例

下面以 GLM-5.1 模型为例。输出分别保存为 JSON，便于比较确定性 token IDs。

### 2.1 DSA 关闭：原生基线非回归

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode disabled \
  --result-json /tmp/dsa-p1-disabled.json
```

预期：正常加载并生成。

### 2.2 DSA 开启：配置与模型能力校验

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode enabled \
  --result-json /tmp/dsa-p1-enabled.json
```

预期：类型化配置解析通过，模型能力识别通过，并正常生成。该结果只说明
P1 控制面工作正常，不说明已经发生稀疏卸载。

### 2.3 不支持组合：拒绝 async scheduling

```bash
python examples/dsa_demo/simple_prompt_test.py \
  --model /home/models/GLM-5.1-W4A8 \
  --mode reject-async \
  --result-json /tmp/dsa-p1-reject-async.json
```

预期：在加载模型权重前失败，脚本识别到以下错误后以成功状态退出：

```text
DSA sparse offload currently requires async_scheduling=False
```

## 3. 结果核对

`disabled` 和 `enabled` 都成功后，核对：

- 两次使用相同模型、prompt、seed 和采样参数；
- 两个 JSON 的 `outputs[*].token_ids` 是否一致；
- `enabled` 日志中是否出现其他配置或模型能力错误；
- `reject-async` 是否准确命中预期错误，而不是在更早位置异常退出。

请保留以下信息用于问题定位：

- 从 `non-default args` 到 KV cache 初始化完成的日志；
- 三份结果 JSON；
- vLLM、vLLM-Ascend commit；
- 模型路径与 architecture；
- CANN、torch、torch-npu 版本。

## 4. 暂未迁移的旧 demo

v0.19 的 `qa_dataset_test.py` 和 `eval_dataset_acc_score.py` 用于完整 DSA
数据面的精度回归。它们将在 P5 接通 Indexer/MLA 解耦、DRAM dump 和
LIDU/KSC/SFA-Offload 后迁移；现在运行这类测试只能测到 v0.23 原生 packed
cache 路径，容易产生错误结论。
