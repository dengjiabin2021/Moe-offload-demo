import torch
import os
import sys
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig

# ------------------ 初始化空壳模型 ------------------
config = Qwen2MoeConfig.from_pretrained("D:\Projects\Moe-offload-demo\models\Qwen1.5-Moe-A2.7B\config.json")
with torch.device('meta'):
    model = Qwen2MoeForCausalLM(config)

# ------------------ 内存与设备检测 ------------------
print("=" * 60)
print("模型创建完成，开始检测资源占用...")
print("=" * 60)

# 1. 统计参数数量与理论内存（如果为真实张量）
total_params = 0
total_bytes = 0
device_counter = {}
for name, param in model.named_parameters():
    total_params += param.numel()
    elem_size = param.element_size()  # 对于 meta 张量，仍然返回 dtype 对应的字节数
    total_bytes += param.numel() * elem_size
    # 记录每个设备出现的次数
    dev = str(param.device)
    device_counter[dev] = device_counter.get(dev, 0) + 1

print(f"参数总数: {total_params:,}")
print(f"若为真实张量，理论内存: {total_bytes / 1024**3:.2f} GB")
print(f"\n参数设备分布: {device_counter}")

# 2. 展示一个参数的具体信息
print("\n示例参数（第一个）:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, device={param.device}, dtype={param.dtype}")
    break

# 3. 检查当前进程的实际内存（可选，需要 psutil）
try:
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"\n当前进程物理内存占用: {mem_info.rss / 1024**2:.2f} MB")
except ImportError:
    print("\n(未安装 psutil，无法显示实际内存，但可观察系统监视器)")

# 4. CUDA 显存占用（应该为 0）
if torch.cuda.is_available():
    print(f"\n当前 GPU 显存分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"当前 GPU 显存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("\n未检测到 GPU，跳过显存统计")

# 5. 验证 meta 张量无法参与计算
print("\n尝试对 meta 张量做加法（预期抛出 RuntimeError）:")
try:
    dummy = next(model.parameters())
    _ = dummy + dummy
except RuntimeError as e:
    print(f"  捕获预期错误: {e}")
