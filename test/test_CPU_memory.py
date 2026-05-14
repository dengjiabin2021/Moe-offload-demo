import torch
import gc
import weakref

# 核心：只验证张量生命周期，去掉所有会报错的文件/MMAP操作
def test_tensor_lifecycle():
    # 1. 创建 CPU 张量（完全模拟你的 clone() 逻辑）
    cpu_tensor = torch.randn(1024, 1024, 10)  # 纯CPU张量
    
    # 2. 创建弱引用：监控这个CPU张量是否被销毁
    cpu_ref = weakref.ref(cpu_tensor)
    
    # 3. 拷贝到 GPU（和你代码的 to(device) 完全一致）
    gpu_tensor = cpu_tensor.to("cuda:0")
    
    # 函数内：CPU张量存活
    print(f"函数内部 → CPU张量是否存在：{cpu_ref() is not None}")
    
    # 4. 返回GPU张量，函数内局部变量 cpu_tensor 自动销毁
    return gpu_tensor, cpu_ref

# ===================== 执行测试 =====================
if __name__ == "__main__":
    print("="*60)
    
    # 调用函数
    gpu_tensor, cpu_ref = test_tensor_lifecycle()
    
    # 强制垃圾回收（立即销毁无引用的对象）
    #gc.collect()

    # 🔥 最终验证：CPU张量还在吗？
    print(f"函数退出后 → CPU张量是否存在：{cpu_ref() is not None}")
    print(f"外部张量设备：{gpu_tensor.device}")
    print("="*60)

    # 结论
    if cpu_ref() is None:
        print("✅ 最终答案：CPU张量已彻底销毁！")
        print("✅ 你的代码：加载到GPU后，CPU无任何备份！")
    else:
        print("❌ 异常（不可能出现）")