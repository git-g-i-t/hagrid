import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= ⚙️ 配置区域 (修改这里) =================

# 1. 定义你的实验名称和对应的路径
# 格式: "图例上显示的名字": "work_dir/实验文件夹名/logs"
# 注意：路径必须指向包含 events.out.tfevents... 文件的那个文件夹
EXPERIMENTS = {
    #"ResNet18_without_pretrained(Baseline)": "work_dir/ResNet18/logs",
    #"ResNet18_with_pretrained": "work_dir/ResNet18_with_pretrained/logs",
    "SE-ResNet18":        "hagrid_v3/work_dir/SE_ResNet18_Attention/logs/train",
    "CBAM-ResNet18":      "hagrid_v3/work_dir/CBAM_ResNet18_Attention/logs/train",
}

# 2. 你想画什么指标？(去 TensorBoard 网页版确认一下 Tag 名字)
# 通常是 "F1Score/Eval" 或 "loss/Train"
TAG_TO_PLOT = "F1Score/Eval" 
# TAG_TO_PLOT = "loss/Train"

# 3. 平滑系数 (0.0 表示不平滑，0.9 表示非常平滑，推荐 0.6-0.8)
SMOOTH_FACTOR = 0.6 

# 4. 图片保存名称
SAVE_NAME = "results/model_comparison_result.png"

# ================= 🔧 工具函数 (不用改) =================

def smooth(scalars, weight):
    """
    平滑曲线函数 (Exponential Moving Average)
    """
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def read_tensorboard_data(log_dir, tag):
    """
    读取 TensorBoard 日志文件
    """
    # 找到该目录下最新的 tfevents 文件
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        print(f"❌ 错误: 在 {log_dir} 找不到日志文件！")
        return None, None
    
    # 选最新的一个日志文件
    event_file = max(event_files, key=os.path.getctime)
    print(f"正在读取: {event_file} ...")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # 检查 Tag 是否存在
    if tag not in ea.Tags()['scalars']:
        print(f"⚠️ 警告: 找不到标签 '{tag}'。可用标签: {ea.Tags()['scalars']}")
        return None, None

    # 提取数据
    events = ea.Scalars(tag)
    steps = [x.step for x in events]
    values = [x.value for x in events]
    
    return steps, values

# ================= 🎨 绘图主逻辑 =================

def main():
    # 设置学术风格
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 颜色库
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'x']

    for i, (label, log_dir) in enumerate(EXPERIMENTS.items()):
        steps, values = read_tensorboard_data(log_dir, TAG_TO_PLOT)
        
        if steps is None or len(steps) == 0:
            continue

        # 数据平滑
        if SMOOTH_FACTOR > 0:
            values = smooth(values, SMOOTH_FACTOR)

        # 绘图
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(steps, values, label=label, color=color, linewidth=2, 
                 marker=marker, markersize=6, markevery=max(1, len(steps)//8))
        
        # 标出最后一个点的值
        plt.text(steps[-1], values[-1], f"{values[-1]:.3f}", fontsize=9, color=color, fontweight='bold')

    # 图表装饰
    plt.title(f"Performance Comparison: {TAG_TO_PLOT}", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(TAG_TO_PLOT, fontsize=12)
    plt.legend(fontsize=10, loc="best", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(SAVE_NAME)
    print(f"\n✅ 绘图完成！已保存为 {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()