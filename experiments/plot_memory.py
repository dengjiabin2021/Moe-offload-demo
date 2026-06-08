import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import sys
# 将项目根目录加入 sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ----------- 数据读取函数 -----------
def read_memory_csv(csv_path: str | Path) -> dict:
    """读取 memory_timeline.csv，返回 {timestamp_s, phase, gpu_alloc, gpu_resv, cpu_rss} 各列。"""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    data: dict[str, list] = {
        "timestamp_s": [],
        "phase":       [],
        "gpu_alloc":   [],
        "gpu_resv":    [],
        "cpu_rss":     [],
    }
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timestamp_s"].append(float(row["timestamp_s"]))
            data["phase"].append(row["phase"])
            data["gpu_alloc"].append(float(row["gpu_allocated_mb"]))
            data["gpu_resv"].append(float(row["gpu_reserved_mb"]))
            data["cpu_rss"].append(float(row["cpu_rss_mb"]))

    return data


# ----------- 参数设置 -----------
csv_file_path = REPO_ROOT / "results/memory_timeline.csv"
save_file_path = REPO_ROOT / "experiments/figures/memory_timeline.png"

line_width_alloc = 3.0
line_width_resv  = 2.4
line_width_cpu   = 2.5
grid_alpha       = 0.15

figsize      = (10, 6.5)
x_label_size = 12
y_label_size = 12
title_size   = 13
x_ticks_size = 10
y_ticks_size = 10
ylabel_pad   = 4
xlabel_pad   = 4

color_gpu_alloc = "#0B4F9C"
color_gpu_resv  = "#6BAED6"
color_cpu       = "#C44E52"

phase_colors = {
    "loading": "#DCF7E2",
    "prefill": "#D6D5F6",
    "decode":  "#EFD8DF"
}
phase_labels = {
    "loading": "Loading",
    "prefill": "Prefill",
    "decode":  "Decode"
}

# 全局字体与负号设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ----------- 数据加载 -----------
# 调用函数读取字典，并转化为 numpy array 方便后续进行 max() 和 argmax() 计算
raw_data = read_memory_csv(csv_file_path)

t         = np.array(raw_data["timestamp_s"])
phases    = raw_data["phase"]
gpu_alloc = np.array(raw_data["gpu_alloc"])
gpu_resv  = np.array(raw_data["gpu_resv"])
cpu_rss   = np.array(raw_data["cpu_rss"])


# ----------- 创建画布与双子图 -----------
fig, (ax_gpu, ax_cpu) = plt.subplots(
    2, 1, 
    figsize=figsize, 
    sharex=True, 
    gridspec_kw={"height_ratios": [2, 1]}
)


# ----------- 折线绘制 (GPU & CPU) -----------
# GPU
ax_gpu.plot(t, gpu_alloc, lw=line_width_alloc, color=color_gpu_alloc)
ax_gpu.plot(t, gpu_resv, lw=line_width_resv, color=color_gpu_resv, alpha=0.9)

# CPU
ax_cpu.plot(t, cpu_rss, lw=line_width_cpu, linestyle="-.", color=color_cpu)


# ----------- 提取阶段分界点 & 绘制背景 -----------
segments = []
start_idx = 0
for i in range(1, len(phases)):
    if phases[i] != phases[i - 1]:
        segments.append((phases[i - 1], t[start_idx], t[i - 1]))
        start_idx = i
segments.append((phases[-1], t[start_idx], t[-1]))

# 在两个子图上绘制背景色块
for ax in [ax_gpu, ax_cpu]:
    ylim = ax.get_ylim()
    for phase_name, t0, t1 in segments:
        ax.axvspan(
            t0, t1, 
            color=phase_colors.get(phase_name, "#EEEEEE"), 
            alpha=0.8, zorder=0
        )
        # 只在 GPU 图表上方标注阶段文本
        if ax == ax_gpu:
            ax.text(
                (t0 + t1) / 2, ylim[1] * 0.96, 
                phase_labels.get(phase_name, phase_name),
                ha="center", va="top", 
                fontsize=10, fontweight="bold", color="#444444"
            )


# ----------- 峰值自动标注 -----------
# GPU 峰值
if len(gpu_alloc) > 0:
    idx_gpu = np.argmax(gpu_alloc)
    px_g, py_g = t[idx_gpu], gpu_alloc[idx_gpu]
    dx_g = -t.max() * 0.15 if px_g > t.max() * 0.7 else t.max() * 0.05

    ax_gpu.annotate(
        f"{py_g / 1024:.2f} GB", 
        xy=(px_g, py_g), xytext=(px_g + dx_g, py_g * 0.92),
        fontsize=10, fontweight="bold", color=color_gpu_alloc,
        arrowprops=dict(arrowstyle="->", color=color_gpu_alloc, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_gpu_alloc, alpha=0.95)
    )

# CPU 峰值
if len(cpu_rss) > 0 and cpu_rss.max() > 0:
    idx_cpu = np.argmax(cpu_rss)
    px_c, py_c = t[idx_cpu], cpu_rss[idx_cpu]
    dx_c = -t.max() * 0.15 if px_c > t.max() * 0.7 else t.max() * 0.05

    ax_cpu.annotate(
        f"{py_c / 1024:.2f} GB", 
        xy=(px_c, py_c), xytext=(px_c + dx_c, py_c * 0.92),
        fontsize=10, fontweight="bold", color=color_cpu,
        arrowprops=dict(arrowstyle="->", color=color_cpu, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_cpu, alpha=0.95)
    )


# ----------- X轴 / Y轴 格式设置 -----------
ax_gpu.set_ylabel("GPU Memory (MB)", fontsize=y_label_size, labelpad=ylabel_pad)
ax_cpu.set_ylabel("CPU Memory (MB)", fontsize=y_label_size, labelpad=ylabel_pad)
ax_cpu.set_xlabel("Time (s)", fontsize=x_label_size, labelpad=xlabel_pad)

for ax in [ax_gpu, ax_cpu]:
    ax.tick_params(axis='y', labelsize=y_ticks_size)
    ax.grid(True, alpha=grid_alpha, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    
    # 设置坐标轴边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        
ax_cpu.tick_params(axis='x', labelsize=x_ticks_size)


# ----------- 图例 -----------
legend_elements = [
    Line2D([0], [0], color=color_gpu_alloc, linewidth=line_width_alloc, label="GPU Allocated"),
    Line2D([0], [0], color=color_gpu_resv, linewidth=line_width_resv, alpha=0.9, label="GPU Reserved"),
    Line2D([0], [0], color=color_cpu, linewidth=line_width_cpu, linestyle="-.", label="CPU RSS")
]

ax_gpu.legend(
    handles=legend_elements,
    fontsize=10,
    loc='center',
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.8
)

# ----------- 标题与脚注 -----------
ax_gpu.set_title("Memory Timeline", pad=28, fontweight="bold", fontsize=title_size)

if len(t) > 0:
    fig.text(
        0.99, 0.01,
        f"Samples={len(t)} | Runtime={t.max():.1f}s | "
        f"GPU Peak={np.max(gpu_alloc)/1024:.2f} GB | CPU Peak={np.max(cpu_rss)/1024:.2f} GB",
        ha="right", fontsize=9, color="gray"
    )

# ----------- 保存与展示 -----------
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# 保存为 PDF 或 PNG (根据需要打开注释)
plt.savefig(save_file_path, format="png", bbox_inches='tight')
plt.show()