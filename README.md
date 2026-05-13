# Moe-offload-demo

先做第一步：扫描本地 safetensors 模型文件，生成每个权重的磁盘位置 manifest。

## 生成权重 manifest

```powershell
python scripts/build_safetensors_manifest.py models/Qwen1.5-Moe-A2.7B
```

默认输出到：

```text
models/Qwen1.5-Moe-A2.7B/weight_manifest.json
```

JSON 顶层就是权重名到位置元数据的映射，不额外写入模型目录、统计信息等字段。

每个权重条目包含：

- `file`: safetensors 分片文件名
- `offset`: 权重数据在文件里的绝对字节偏移
- `nbytes`: 权重数据字节数
- `dtype`: safetensors 记录的数据类型
- `shape`: tensor 形状

## 按需加载 MoE 专家 demo

先安装依赖：

```powershell
pip install -r requirements.txt
```

然后运行：

```powershell
python scripts/qwen_moe_ondemand_demo.py --model-dir models/Qwen1.5-Moe-A2.7B
```

当前 demo 的行为：

- 初始化模型时使用 meta 初始化，避免随机初始化参数。
- 非 routed expert 权重和 shared expert 权重会加载到 GPU 并常驻。
- `model.layers.*.mlp.experts.*` 的 routed expert 权重不会在初始化阶段加载。
- forward 命中某个 expert 时，按 tensor_name 从 `weight_manifest.json` 精确读取该 expert 的 `gate_proj/up_proj/down_proj`。
- expert 计算完成后删除临时引用，并默认调用 `torch.cuda.empty_cache()`。
