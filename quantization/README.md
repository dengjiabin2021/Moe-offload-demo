# MoE 路由专家 FP8 量化

本目录只保留两个 Python 脚本，不修改 `moe_runtime/` 或原始模型。

| 脚本 | 用途 |
| --- | --- |
| `quantize.py` | 用 GPTQ 量化路由专家并保存模型 |
| `test_quantized.py` | 检查 FP8 权重与 scale，再加载模型生成文本 |

保留 `calibration.json` 和 `outputs/`；脚本不会自动删除工作缓存或历史产物。
依赖使用项目根目录 `requirements.txt`，在独立的 `moe-offload` 环境中运行。

## 量化

当前支持本地 `deepseek_v2` 和 `qwen2_moe` 模型。
`build_target_names()` 按各自配置选择专家层和投影，排除 dense MLP、共享专家、router 和 attention。
适配其他架构时，在这个函数中增加明确分支，不自动猜测专家命名。

方案为 GPTQ + `FP8_DYNAMIC`：权重逐通道静态 FP8，输入逐 token 动态 FP8，scale 使用 FP32。
保留加载后拆分专家的流程；`sequential_targets` 只选择待量化的专家投影，attention 整块执行，
不展开其 RoPE/KV 计算的 FX 跟踪，也不把整层所有专家作为一个 GPU 校准单元。

在项目根目录运行 DeepSeek 量化：

```powershell
conda activate moe-offload
python -u quantization/quantize.py --cpu-memory 5GiB --num-samples 256 --max-seq-length 512
```

固定输出为 `quantization/outputs/DeepSeek-V2-Lite-GPTQ-FP8`，不使用时间戳。
输出目录已存在时直接报错，不会覆盖或清理；CPU 预算只约束权重驻留，不包含激活和 Hessian。
加载和校准仍需要磁盘工作空间，不能只预留最终 FP8 模型的大小。

切换 Qwen：

```powershell
python -u quantization/quantize.py --model-dir models/Qwen1.5-Moe-A2.7B --cpu-memory 5GiB
```

输出会固定为 `quantization/outputs/Qwen1.5-Moe-A2.7B-GPTQ-FP8`，也可通过 `--output-dir` 指定。
`--dry-run` 只构建模型目标与配方，不加载权重、不创建输出目录。

默认校准文件是本目录的 `calibration.json`，同时支持 JSON 数组和 JSONL。
文本字段默认 `text`；其他字段通过 `--text-column` 指定，例如：

```powershell
python -u quantization/quantize.py --calibration-file quantization/calibration.json --text-column text
```

## 运行检查

```powershell
python -u quantization/test_quantized.py --max-new-tokens 16
```

测试其他量化产物：

```powershell
python -u quantization/test_quantized.py --model-dir quantization/outputs/Qwen1.5-Moe-A2.7B-GPTQ-FP8
```

先检查最终分片里的二维 FP8 权重、FP32 scale 和动态 A8 配置，再加载并检查生成 token、logits 和文本。
默认 GPU 权重预算 `6GiB`、CPU 权重预算 `5GiB`，可通过 `--gpu-memory` 和 `--cpu-memory` 调整。
测试时使用 Transformers 的 BF16 反量化路径，额外工作缓存位于产物目录下的 `test_offload/`。
这是量化文件的加载与生成功能检查，不是原生 FP8 W8A8 kernel 测速，也不代表精度评估通过。

本次仍不接入项目自己的卸载或预取运行时；那部分需要另外适配 FP8 权重、scale 和计算内核。
