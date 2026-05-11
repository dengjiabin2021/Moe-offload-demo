# Moe-offload-demo

一个用于演示 MoE 专家权重按需卸载、CPU/GPU 二级缓存和路由预取的最小项目。

第一版复用 Transformers 的 Qwen2-MoE 实现路径，目标适配本地
`models/Qwen1.5-MoE-A2.7B`。非专家权重默认常驻目标设备，专家权重在 forward
时按 router 选中结果加载。

## Quick start

```powershell
& "D:\Anaconda\envs\py3.11_torch_cu12.4\python.exe" examples/qwen2_moe_generate.py `
  --model-path models/Qwen1.5-MoE-A2.7B `
  --gpu-cache 2GiB `
  --cpu-cache 8GiB
```

```powershell
& "D:\Anaconda\envs\py3.11_torch_cu12.4\python.exe" examples/phimoe_generate.py `
  --model-path models/Phi-tiny-MoE-instruct `
  --gpu-cache 1GiB `
  --cpu-cache 4GiB
```

## Run tests

```powershell
& "D:\Anaconda\envs\py3.11_torch_cu12.4\python.exe" -m pytest tests
```

## Layout

- `moe_runtime/core/`: 模型无关的缓存、索引、专家执行和配置。
- `moe_runtime/adapters/qwen2_moe/`: Qwen2-MoE / Qwen1.5-MoE checkpoint 适配。
- `moe_runtime/adapters/phimoe/`: Phi-tiny-MoE / PhiMoE checkpoint 适配。
- `moe_runtime/integrations/`: Transformers 加载入口。
- `examples/`: 可运行 demo。
- `tests/`: 单元测试和 tiny checkpoint 集成测试。
