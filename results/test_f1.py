import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= ⚙️ 配置区域 =================

EXPERIMENTS = {
    "Baseline":        "hagrid_v2/work_dir/ResNet18_base/logs/test",
    "Pretrained":      "hagrid_v2/work_dir/ResNet18_pre/logs/test",
    "SE-ResNet18":     "hagrid_v2/work_dir/SE_ResNet18_Attention/logs/test",
    "CBAM-ResNet18":   "hagrid_v2/work_dir/CBAM_ResNet18_Attention/logs/test",
    "Coord-ResNet18":  "hagrid_v2/work_dir/Coord_ResNet18/logs/test",
}

# 确保这里的 Tag 名字和 TensorBoard 中完全一致
TAG_TO_PLOT = "F1Score/Test" 

SAVE_NAME = "results/test_f1_final_comparison.png"

# ================= 🔧 工具函数 =================

def get_test_score(log_dir, tag):
    """提取唯一的测试集分数"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        print(f"❌ 错误: {log_dir} 没找到日志文件")
        return None
    
    event_file = max(event_files, key=os.path.getctime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        # 自动尝试可能的变体
        for alt_tag in ["F1/Test", "Test/F1Score", "Test_F1"]:
            if alt_tag in ea.Tags()['scalars']:
                tag = alt_tag
                break
        else:
            print(f"⚠️ 警告: 找不到标签 '{tag}'。可用标签: {ea.Tags()['scalars']}")
            return None
    
    events = ea.Scalars(tag)
    # 取最后一个值（即使只有一轮，也是最后一个）
    return events[-1].value

# ================= 🎨 绘图主逻辑 =================

def main():
    names = []
    scores = []

    # 1. 收集数据
    for name, path in EXPERIMENTS.items():
        score = get_test_score(path, TAG_TO_PLOT)
        if score is not None:
            names.append(name)
            scores.append(score)

    if not scores:
        print("❌ 未能提取到任何有效数据，请检查路径或标签名。")
        return

    # 2. 绘图设置 (白色背景)
    plt.style.use('default') # 回归标准简洁风格
    plt.rcParams['figure.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 颜色组合：选择比较清爽的学术配色
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#9013FE']
    
    # 3. 绘制柱状图
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, scores, color=colors[:len(names)], 
                  width=0.5, edgecolor='#333333', linewidth=1.2)

    # 4. 在柱子上方标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='black')

    # 5. 细节修饰
    ax.set_title("Test Set Performance Comparison (F1-Score)", fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11, fontweight='medium')
    
    # 设置 Y 轴范围：从 0 开始，稍微高于最高分
    max_s = max(scores)
    ax.set_ylim(0, min(1.0, max_s + 0.15)) 
    
    # 只开启 Y 轴网格线，设为浅灰色虚线
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True) # 让网格线在柱子下方

    # 去掉上方和右方的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 6. 保存输出
    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_NAME), exist_ok=True)
    plt.savefig(SAVE_NAME, bbox_inches='tight')
    print(f"\n✅ 绘图完成！图片保存至: {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()