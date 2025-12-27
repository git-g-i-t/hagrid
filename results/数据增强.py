import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= ⚙️ 配置区域 =================

# 定义实验名称和对应的根目录 (脚本会自动寻找其下的 logs/train 和 logs/test)
EXPERIMENTS = {
    "ResNet18 (Baseline)":   "hagrid_v2/work_dir/ResNet18_base",
    "ResNet18 (数据增强)": "hagrid_v2/work_dir/ResNet18",
}

# 标签定义
TAG_LOSS_TRAIN = "loss/Train"    # 训练损失
TAG_F1_VAL     = "F1Score/Eval"  # 验证集 F1 (从 train 日志读取)
TAG_F1_TEST    = "F1Score/Test"  # 测试集 F1 (从 test 日志读取)

# 平滑系数
SMOOTH_FACTOR = 0.6
SAVE_NAME = "results/resnet18_pretrain_comparison.png"

# ================= 🔧 工具函数 =================

def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def read_tb_scalar(log_dir, tag):
    """从目录下读取指定的 TensorBoard 标签数据"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files: return None, None
    
    event_file = max(event_files, key=os.path.getctime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']: return None, None
    
    events = ea.Scalars(tag)
    return [x.step for x in events], [x.value for x in events]

# ================= 🎨 绘图主逻辑 =================

def main():
    # 设置全局样式：白色背景
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 创建 3行1列 的画布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), dpi=150)
    
    colors = ['#1f77b4', '#ff7f0e'] # 蓝，橙
    markers = ['o', 's']
    
    test_names = []
    test_values = []

    for i, (label, base_path) in enumerate(EXPERIMENTS.items()):
        color = colors[i]
        marker = markers[i]
        
        # --- 1. 读取并绘制 Training Loss (从 logs/train) ---
        train_log = os.path.join(base_path, "logs/train")
        steps, loss = read_tb_scalar(train_log, TAG_LOSS_TRAIN)
        if loss:
            plot_loss = smooth(loss, SMOOTH_FACTOR)
            ax1.plot(steps, plot_loss, label=label, color=color, linewidth=2)
            min_l = min(plot_loss)
            ax1.annotate(f"Min: {min_l:.3f}", xy=(steps[np.argmin(plot_loss)], min_l), 
                         xytext=(5, 5), textcoords='offset points', color=color, fontweight='bold', fontsize=9)

        # --- 2. 读取并绘制 Val F1 (从 logs/train) ---
        steps, f1_val = read_tb_scalar(train_log, TAG_F1_VAL)
        if f1_val:
            plot_f1 = smooth(f1_val, SMOOTH_FACTOR)
            ax2.plot(steps, plot_f1, label=label, color=color, linewidth=2, marker=marker, markevery=max(1, len(steps)//10))
            max_f1 = max(plot_f1)
            ax2.annotate(f"Best: {max_f1:.3f}", xy=(steps[np.argmax(plot_f1)], max_f1), 
                         xytext=(5, -15), textcoords='offset points', color=color, fontweight='bold', fontsize=9)

        # --- 3. 读取 Test F1 (从 logs/test) ---
        test_log = os.path.join(base_path, "logs/test")
        _, f1_test = read_tb_scalar(test_log, TAG_F1_TEST)
        if f1_test:
            test_names.append(label)
            test_values.append(f1_test[-1]) # 取最后一轮测试值

    # --- 完善 Subplot 1 (Loss) ---
    ax1.set_title("Training Loss Convergence", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 完善 Subplot 2 (Val F1) ---
    ax2.set_title("Validation F1 Score during Training", fontsize=14, fontweight='bold')
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    # --- 完善 Subplot 3 (Test F1 - 柱状图) ---
    bars = ax3.bar(test_names, test_values, color=colors, width=0.4, edgecolor='black', linewidth=1)
    ax3.set_title("Final Test Set Performance", fontsize=14, fontweight='bold')
    ax3.set_ylabel("F1 Score", fontsize=12)
    ax3.set_ylim(0, 1.1)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}', ha='center', fontweight='bold')

    # 整体布局调整
    plt.tight_layout(pad=4.0)
    

    os.makedirs(os.path.dirname(SAVE_NAME), exist_ok=True)
    plt.savefig(SAVE_NAME)
    print(f"✅ 对比图已保存至: {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()