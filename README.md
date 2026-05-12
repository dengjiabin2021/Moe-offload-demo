# Moe-offload-demo

先做第一步：扫描本地 safetensors 模型文件，生成每个权重的磁盘位置 manifest。

## 生成权重 manifest

```powershell
python scripts/build_safetensors_manifest.py --model-dir models/Qwen1.5-Moe-A2.7B
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
