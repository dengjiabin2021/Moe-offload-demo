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

## 当前架构

项目分成三层：

- `moe_runtime/Loader`: 底层 tensor 读取，通过 tensor name 从 manifest 精确加载单个 tensor。
- `moe_runtime/core`: 通用 MoE 加载框架，只负责 meta 初始化、常驻权重加载、expert 权重跳过和校验。
- `moe_runtime/adapters`: 模型适配层，目前实现 `Qwen15MoeA27BAdapter`。

后续适配其他 MoE 模型时，优先新增 Adapter，不修改通用 Loader。

## Qwen1.5-MoE-A2.7B 加载测试

先安装依赖：

```powershell
pip install -r requirements.txt
```

然后运行加载测试：

```powershell
python test/test_load_model.py
```

当前阶段只实现加载逻辑，不做推理：

- 初始化模型时使用 meta 初始化，避免随机初始化参数。
- `ResidentWeightLoader` 是通用流程，不写 Qwen 专属逻辑。
- `Qwen15MoeA27BAdapter` 负责识别 Qwen 的 routed expert 命名。
- 非 routed expert 权重和 shared expert 权重会加载到 GPU 并常驻。
- routed expert 权重不会在初始化阶段加载，模型里的专家参数保持在 meta 设备。
- 加载完成后会校验：除了 adapter 允许的 routed expert 参数，其他参数不允许仍留在 meta 上。
