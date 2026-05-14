import gc
import sys
from pathlib import Path

# 当前文件路径配置
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_runtime import Qwen15MoeA27BAdapter, ResidentWeightLoader, get_parameter_device_stats

# 打印分设备统计结果
def print_device_stats(stats, title):
    print(f"\n[统计] {title}")
    print(f"Meta设备 参数: {stats['meta']['params']:,} | 占用: {stats['meta']['mb']} MB")
    print(f"CPU设备  参数: {stats['cpu']['params']:,} | 占用: {stats['cpu']['mb']} MB")
    print(f"GPU设备  参数: {stats['cuda']['params']:,} | 占用: {stats['cuda']['mb']} MB")
    total_params = stats['meta']['params'] + stats['cpu']['params'] + stats['cuda']['params']
    total_mb = stats['meta']['mb'] + stats['cpu']['mb'] + stats['cuda']['mb']
    print(f"[总计] 参数: {total_params:,} | 总计占用: {round(total_mb, 2)} MB")

def main() -> None:
    model_dir = "models/Qwen1.5-Moe-A2.7B"
    manifest_path = "models/Qwen1.5-Moe-A2.7B/weight_manifest.json"
    device = "cuda"

    # ============== 测试：通过通用 Loader + Qwen Adapter 加载 ==============
    print("\n" + "="*80)
    print("[测试] 通用 ResidentWeightLoader + Qwen15MoeA27BAdapter")
    loader = ResidentWeightLoader(adapter=Qwen15MoeA27BAdapter())
    model = loader.load(
        model_dir=model_dir,
        manifest_path=manifest_path,
        device=device,
    )
    stats = get_parameter_device_stats(model)
    print_device_stats(stats, "常驻权重加载 设备分布")
    print(f"加载报告: {model.moe_load_report}")
    del model
    gc.collect()

if __name__ == "__main__":
    main()
