import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= ⚙️ 配置区域 =================

EXPERIMENTS = {
    "Baseline":        "hagrid_v2/work_dir/ResNet18_base/logs/train",
    "Pretrained":      "hagrid_v2/work_dir/ResNet18_pre/logs/train",
    "SE-ResNet18":     "hagrid_v2/work_dir/SE_ResNet18_Attention/logs/train",
    "CBAM-ResNet18":   "hagrid_v2/work_dir/CBAM_ResNet18_Attention/logs/train",
    "Coord-ResNet18":  "hagrid_v2/work_dir/Coord_ResNet18/logs/train",
}

TAG_TO_PLOT = "loss/Train" 
SMOOTH_FACTOR = 0.6  # 稍微平滑，保留一定的波动细节
SAVE_NAME = "results/loss_convergence_white.png"

# ================= 🔧 工具函数 =================

def smooth(scalars, weight):
    if weight <= 0: return scalars
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def read_tensorboard_data(log_dir, tag):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        print(f"❌ 错误: {log_dir} 没找到日志")
        return None, None
    event_file = max(event_files, key=os.path.getctime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return None, None
    events = ea.Scalars(tag)
    return [x.step for x in events], [x.value for x in events]

# ================= 🎨 绘图主逻辑 =================

def main():
    # 1. 设置白色背景风格
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    
    # 颜色和标记符号
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v'] # 圆形、正方形、上三角、菱形、下三角

    for i, (label, log_dir) in enumerate(EXPERIMENTS.items()):
        steps, values = read_tensorboard_data(log_dir, TAG_TO_PLOT)
        if steps is None: continue

        # 平滑处理
        plot_values = smooth(values, SMOOTH_FACTOR)
        
        # 寻找最小值点
        min_val = min(plot_values)
        min_idx = np.argmin(plot_values)
        min_step = steps[min_idx]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 绘制折线
        # markevery: 每隔几个点画一个形状，防止点太密集
        ax.plot(steps, plot_values, label=label, color=color, linewidth=2,
                marker=marker, markersize=7, markevery=max(1, len(steps)//15),
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)

        # 仅标注最小值
        # xytext 稍微向右偏移一点，防止遮挡折线
        ax.annotate(f"{min_val:.3f}", 
                    xy=(min_step, min_val), 
                    xytext=(5, 2), 
                    textcoords='offset points',
                    fontsize=10, 
                    color=color, 
                    fontweight='bold')

    # 图表细节装饰
    ax.set_title("Training Loss Convergence", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("Train", fontsize=13)
    
    # 设置网格为灰色虚线
    ax.grid(True, linestyle='--', color='lightgray', alpha=0.8)
    
    # 去掉上方和右方的边框（可选，让画面更开阔）
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_NAME), exist_ok=True)
    plt.savefig(SAVE_NAME, bbox_inches='tight')
    print(f"\n✅ 绘图完成！图片已保存至: {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()