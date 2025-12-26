"""
主运行脚本 (run.py)
功能：
1. 解析命令行参数（训练/测试模式、配置文件路径、GPU数量）。
2. 初始化分布式训练环境（如果使用多 GPU）。
3. 加载模型、数据集加载器 (DataLoaders)、优化器。
4. 根据任务类型（检测 vs 分类）选择评估指标（mAP vs F1-Score）。
5. 初始化 Trainer 管理器并执行训练或测试循环。
"""
import argparse
from typing import Optional, Tuple

# 配置管理工具
from omegaconf import OmegaConf

# 尝试导入目标检测的评估指标：Mean Average Precision (mAP)
# try-except 块是为了兼容不同版本的 torchmetrics 库（旧版本可能叫 MAP）
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP

# 导入自定义工具模块

# train_utils: 包含加载数据、加载优化器以及核心的 Trainer 类
from custom_utils.train_utils import Trainer, load_train_objects, load_train_optimizer
# utils: 自定义的带日志功能的 F1 分数计算器（用于分类任务）
from custom_utils.utils import F1ScoreWithLogging


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Gesture classifier...")

    # -c / --command: 可选参数，指定运行模式 ，choices=("train", "test") 限制只能输入 train 或 test (默认值为 train)
    parser.add_argument(
        "-c", "--command", required=False, type=str, default="train", help="Training or test pipeline", choices=("train", "test")
    )
    # -p / --path_to_config: 可选参数，YAML 配置文件的路径默认(configs/ResNet18.yaml)
    parser.add_argument("-p", "--path_to_config", required=False, type=str, default="hagrid_v2\configs\se_resnet18.yaml   ", help="Path to config")
    # --n_gpu: 可选参数，指定使用的 GPU 数量，默认为 1
    parser.add_argument("--n_gpu", required=False, type=int, default=1, help="Number of GPUs to use")

    known_args, _ = parser.parse_known_args(params)
    return known_args


def run(args):
    """
    主执行函数
    """
    # 1. 加载 YAML 配置文件
    config = OmegaConf.load(args.path_to_config)

    # 3. 加载训练所需的关键对象
    # train_dataloader: 训练数据加载器
    # val_dataloader: 验证数据加载器
    # test_dataloader: 测试数据加载器
    # model: 初始化的深度学习模型架构
    train_dataloader, val_dataloader, test_dataloader, model = load_train_objects(config, args.command, args.n_gpu)

    # 4. 根据模型类型选择评估指标 (Metric)
    if model.type == "detector":
        # 如果是目标检测模型（如 SSD, YOLO），使用 mAP (Mean Average Precision)
        metric = MeanAveragePrecision(class_metrics=False)  # class_metrics=False 表示只计算整体 mAP，不分开打印每个类别的 AP
    else:
        # 如果是分类模型
        # 根据配置判断是二分类 (binary) 还是多分类 (multiclass)
        task = "binary" if config.dataset.one_class else "multiclass"
        # 确定类别数量
        num_classes = 2 if config.dataset.one_class else len(config.dataset.targets)
        # 使用自定义的 F1 分数计算器
        metric = F1ScoreWithLogging(task=task, num_classes=num_classes)

    # 5. 加载优化器 (Optimizer) 和 学习率调度器 (Scheduler)
    optimizer, scheduler = load_train_optimizer(model, config)

    # 6. 初始化 Trainer 实例
    # Trainer 类封装了具体的训练循环（Forward, Backward, Loss计算）和验证逻辑
    
    # 确定日志子目录 (train 或 test)
    log_subdir = "train" if args.command == "train" else "test"
    
    trainer = Trainer(
        model=model,                    # 模型
        config=config,                  # 配置
        optimizer=optimizer,            # 优化器
        scheduler=scheduler,            # 调度器
        metric_calculator=metric,       # 评估指标计算器
        train_data=train_dataloader,    # 训练数据
        val_data=val_dataloader,        # 验证数据
        test_data=test_dataloader,      # 测试数据
        n_gpu=args.n_gpu,               # GPU 数量
        log_subdir=log_subdir,          # 日志子目录 (新增)
    )

    # 7. 根据命令行参数执行相应操作
    if args.command == "train":
        # 开始训练流程（包含训练循环和验证循环）
        trainer.train()

    if args.command == "test":
        # 仅执行测试流程（使用 test_dataloader 评估模型)
        trainer.test()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
