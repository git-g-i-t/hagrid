import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Validation vs Test Performance")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to TensorBoard log directory (containing events.out.tfevents...)")
    parser.add_argument("--test_score", type=float, default=None, help="Manual Final Test F1 Score (optional, draw as dashed line)")
    parser.add_argument("--output", type=str, default="results/val_test_curve.png", help="Output image path")
    return parser.parse_args()

def read_tensorboard(log_dir, tag):
    """
    读取 TensorBoard 日志中的标量数据
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
        print(f"⚠️ 提示: 找不到标签 '{tag}'。")
        return None, None

    # 提取数据
    events = ea.Scalars(tag)
    steps = [x.step for x in events]
    values = [x.value for x in events]
    
    return steps, values

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 读取验证集数据
    val_steps, val_scores = read_tensorboard(args.log_dir, "F1Score/Eval")
    
    # 读取测试集数据 (如果有的话)
    test_steps, test_scores = read_tensorboard(args.log_dir, "F1Score/Test")
    
    if not val_steps and not test_steps:
        print("❌ 没有找到任何 F1Score 数据，无法绘图。")
        return

    # 开始绘图
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use('ggplot')
    
    # 绘制验证集曲线
    if val_steps:
        plt.plot(val_steps, val_scores, label="Validation (Eval)", color="#1f77b4", linewidth=2, marker='o')
        # 标注最大值
        max_val = max(val_scores)
        max_idx = val_scores.index(max_val)
        plt.annotate(f'Best Val: {max_val:.3f}', 
                     xy=(val_steps[max_idx], max_val), 
                     xytext=(val_steps[max_idx], max_val + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center')

    # 绘制测试集曲线 (如果有)
    if test_steps:
        plt.plot(test_steps, test_scores, label="Test", color="#d62728", linewidth=2, marker='s')
    
    # 如果手动提供了测试集分数 (画虚线)
    if args.test_score is not None:
        plt.axhline(y=args.test_score, color='#d62728', linestyle='--', label=f'Final Test Score: {args.test_score:.3f}')

    plt.title("Validation vs Test Performance (F1-Score)", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"\n✅ 绘图完成！已保存为 {args.output}")
    # plt.show() # 服务器环境通常不显示

if __name__ == "__main__":
    main()
