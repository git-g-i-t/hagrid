"""
主运行脚本 (MindSpore 适配版)
"""
import argparse
import os

if 'MS_ENABLE_TFT' in os.environ:
    del os.environ['MS_ENABLE_TFT']
os.environ['GLOG_v'] = '3'        # 只显示 ERROR 日志，屏蔽 INFO/WARNING

import mindspore as ms
from omegaconf import OmegaConf

# 导入自定义工具模块 (根据你的文件结构)
from custom_utils.train_utils import Trainer, load_train_objects, load_train_optimizer
from custom_utils.utils import F1ScoreWithLogging, set_random_seed

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="MindSpore Gesture Classifier")

    # 指定运行模式: train 或 test
    parser.add_argument(
        "-c", "--command", required=False, type=str, default="train", 
        help="Training or test pipeline", choices=("train", "test")
    )
    # 配置文件路径
    parser.add_argument(
        "-p", "--path_to_config", required=False, type=str, 
        default="hagrid_v3/configs/se_resnet18_ms.yaml", help="Path to config"
    )
    
    known_args, _ = parser.parse_known_args()
    return known_args

def run(args):
    """
    主执行函数
    """
    # 1. 设置 MindSpore 运行上下文
    # 既然是 Windows CPU 环境，设置为 GRAPH_MODE 以获得更好的性能
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device("CPU")

    # 2. 加载 YAML 配置文件
    config = OmegaConf.load(args.path_to_config)
    
    # 3. 设置随机种子 (保持实验可复现)
    set_random_seed(seed=42)

    # 4. 加载训练对象 (DataLoaders, Model)
    # 注意：MindSpore 不需要传 n_gpu 参数
    train_dataloader, val_dataloader, test_dataloader, model = load_train_objects(config, args.command)

    # 5. 设置评价指标 (F1-Score)
    # 根据配置判断类别数量
    num_classes = len(config.dataset.targets)
    # 适配多分类任务
    metric = F1ScoreWithLogging(task="multiclass", num_classes=num_classes)

    # 6. 加载优化器 (内部已自动将 AdamW 映射为 AdamWeightDecay)
    optimizer, scheduler = load_train_optimizer(model, config)

    # 7. 绑定损失函数 (关键：适配 MindSpore 风格)
    # 对应 PyTorch 的 CrossEntropyLoss，sparse=True 表示输入是 label 索引
    model.criterion = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 8. 确定日志子目录
    log_subdir = "train" if args.command == "train" else "test"
    
    # 9. 初始化 Trainer
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_calculator=metric,
        train_data=train_dataloader,
        val_data=val_dataloader,
        test_data=test_dataloader,
        log_subdir=log_subdir,
    )

    # 10. 执行流程
    if args.command == "train":
        print(f"开始训练: {config.experiment_name} | Epochs: {config.epochs}")
        trainer.train()

    if args.command == "test":
        print(f"开始测试: {config.experiment_name}")
        trainer.test()

if __name__ == "__main__":
    args = parse_arguments()
    run(args)