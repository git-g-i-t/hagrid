import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= ⚙️ 配置区域 (修改这里) =================

# 1. 定义你的实验名称和对应的路径
EXPERIMENTS = {
    "ResNet18_without_pretrained(Baseline)": "hagrid_v2/work_dir/ResNet18_base/logs/train",
    #"ResNet18_with_Data_Augmentation(Baseline_1)": "hagrid_v2/work_dir/ResNet18/logs/train",
    "ResNet18_with_pretrained": "hagrid_v2/work_dir/ResNet18_pre/logs/train",
    "SE-ResNet18":        "hagrid_v2/work_dir/SE_ResNet18_Attention/logs/train",
    "CBAM-ResNet18":      "hagrid_v2/work_dir/CBAM_ResNet18_Attention/logs/train",
}

# 2. 你想画什么指标？
TAG_TO_PLOT = "F1Score/Eval" 
#TAG_TO_PLOT = "loss/Train"

# 3. 平滑系数
SMOOTH_FACTOR = 0

# 4. 图片保存名称
if "loss" in TAG_TO_PLOT.lower():
    SAVE_NAME = "results/model_loss_comparison.png"
else:
    SAVE_NAME = "results/model_f1_comparison_smoth.png"

# ================= 🔧 工具函数 (不用改) =================

def smooth(scalars, weight):
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
        print(f"❌ 错误: 在 {log_dir} 找不到日志文件！")
        return None, None
    
    event_file = max(event_files, key=os.path.getctime)
    print(f"正在读取: {event_file} ...")

    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        print(f"⚠️ 警告: 找不到标签 '{tag}'。可用标签: {ea.Tags()['scalars']}")
        return None, None

    events = ea.Scalars(tag)
    steps = [x.step for x in events]
    values = [x.value for x in events]
    
    return steps, values

# ================= 🎨 绘图主逻辑 =================

def main():
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6), dpi=150)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'x']

    for i, (label, log_dir) in enumerate(EXPERIMENTS.items()):
        steps, values = read_tensorboard_data(log_dir, TAG_TO_PLOT)
        
        if steps is None or len(steps) == 0:
            continue

        # 数据平滑
        plot_values = smooth(values, SMOOTH_FACTOR) if SMOOTH_FACTOR > 0 else values

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # 绘图
        plt.plot(steps, plot_values, label=label, color=color, linewidth=2, 
                 marker=marker, markersize=6, markevery=max(1, len(steps)//8))
        
        # 找到最值 (loss找最小，f1找最大)
        if "loss" in TAG_TO_PLOT.lower():
            target_val = min(values)
            target_idx = values.index(target_val)
            target_step = steps[target_idx]
            
            # 标出最低点的值
            plt.text(target_step, target_val, f"{target_val:.3f}", fontsize=9, color=color, fontweight='bold', ha='center', va='top')
            # 在最低点画一个特别的标记
            plt.plot(target_step, target_val, 'v', color=color, markersize=8, markeredgecolor='white', zorder=10)
        else:
            target_val = max(values)
            target_idx = values.index(target_val)
            target_step = steps[target_idx]

            # 标出最高点的值
            plt.text(target_step, target_val, f"{target_val:.3f}", fontsize=9, color=color, fontweight='bold', ha='center', va='bottom')
            # 在最高点画一个特别的标记
            plt.plot(target_step, target_val, 'o', color=color, markersize=8, markeredgecolor='white', zorder=10)

    plt.title(f"Performance Comparison: {TAG_TO_PLOT}", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(TAG_TO_PLOT, fontsize=12)
    plt.legend(fontsize=10, loc="best", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 稍微拉高 y 轴上限，防止最高点的文字超出边界
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max + (y_max - y_min) * 0.1)

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(SAVE_NAME)):
        os.makedirs(os.path.dirname(SAVE_NAME), exist_ok=True)
        
    plt.savefig(SAVE_NAME)
    print(f"\n✅ 绘图完成！已保存为 {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()