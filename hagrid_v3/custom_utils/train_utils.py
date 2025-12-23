import os
from typing import List, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from custom_utils.utils import Logger, build_model, get_transform
from models import HaGRIDModel

# 设置随机种子，确保实验可复现
from .utils import set_random_seed

set_random_seed()


def get_available_device():
    """
    获取可用设备，优先使用 GPU，如果没有则使用 CPU
    """
    if torch.cuda.is_available():
        return "cuda:0"  # 使用第一个 GPU
    else:
        return "cpu"


def collate_fn(batch: List) -> Tuple:
    """
    DataLoader 的整理函数
    默认的 collate_fn 尝试将所有数据堆叠(stack)成 Tensor，但这对于目标检测不适用，
    因为每张图的 BBox 数量不同。这里只是简单地将 batch 打包成元组列表。

    Parameters
    ----------
    batch : List
        [ (img1, target1), (img2, target2), ... ]
    """
    return list(zip(*batch))


def get_dataloader(dataset: Dataset, **kwargs) -> DataLoader:
    """
    构建 PyTorch DataLoader 的工厂函数

    Parameters
    ----------
    dataset : Dataset
        数据集实例
    **kwargs
        包含 batch_size, num_workers 等参数
    """
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=kwargs["shuffle"],
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
        prefetch_factor=kwargs["prefetch_factor"],
    )


def load_train_objects(config: DictConfig, command: str, n_gpu: int):
    """
    核心工厂函数：加载训练所需的所有对象 (数据、模型)
    根据配置文件自动判断是加载检测任务还是分类任务的 Dataset。

    Parameters
    ----------
    config : DictConfig
        全局配置
    command : str [train, test]
        当前模式
    n_gpu : int
        GPU 数量（保留参数，但实际只支持单设备）

    Returns
    -------
    Tuple
        (train_loader, val_loader, test_loader, model)
    """
    # 1. 构建模型结构
    model = build_model(config)

    # 2. 根据模型类型选择对应的数据集类
    if model.type == "detector":
        from dataset import DetectionDataset as GestureDataset
    elif model.type == "classifier":
        from dataset import ClassificationDataset as GestureDataset
    else:
        raise Exception(f"Model type {model.type} does not exist")

    # 3. 初始化测试集 (无论训练还是测试模式都需要)
    test_dataset = GestureDataset(config, "test", get_transform(config.test_transforms, model.type))

    # 4. 如果是训练模式，初始化训练集和验证集
    if command == "train":
        train_dataset = GestureDataset(config, "train", get_transform(config.train_transforms, model.type))
        if config.dataset.dataset_val and config.dataset.annotations_val:
            val_dataset = GestureDataset(config, "val", get_transform(config.val_transforms, model.type))
        else:
            raise Exception("Cannot train without validation data")
    else:
        train_dataset = None
        val_dataset = None

    # 5. 构建 DataLoaders
    test_dataloader = get_dataloader(test_dataset, **config.test_params)
    if command == "train":
        train_dataloader = get_dataloader(train_dataset, **config.train_params)
        if val_dataset:
            val_dataloader = get_dataloader(val_dataset, **config.val_params)
        else:
            val_dataloader = None
    else:
        # 测试模式不需要训练/验证加载器
        train_dataloader = None
        val_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader, model


def load_train_optimizer(model: HaGRIDModel, config: DictConfig):
    """
    加载优化器 (Optimizer) 和 学习率调度器 (Scheduler)
    """
    # 过滤掉不需要梯度的参数 (例如冻结的主干网络)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    
    # 动态获取优化器类 (如 torch.optim.SGD) 并初始化
    optimizer = getattr(torch.optim, config.optimizer.name)(parameters, **config.optimizer.params)
    
    # 动态获取调度器类 (如 torch.optim.lr_scheduler.StepLR) 并初始化
    if hasattr(config, 'scheduler') and hasattr(config.scheduler, 'name') and config.scheduler.name:
        try:
            scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)
        except AttributeError:
            print(f"⚠️  调度器 {config.scheduler.name} 不存在，使用默认 StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    return optimizer, scheduler


class Trainer:
    """
    训练总管：管理训练循环、验证、保存模型等所有逻辑
    """
    def __init__(
        self,
        model: HaGRIDModel,
        config: DictConfig,
        test_data: torch.utils.data.DataLoader,
        train_data: torch.utils.data.DataLoader = None,
        val_data: torch.utils.data.DataLoader = None,
        metric_calculator=None,
        n_gpu: int = 1,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        log_subdir: str = None,  # 新增参数：日志子目录
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        
        # 设备设置
        self.device = get_available_device()
        print(f"🎯 使用设备: {self.device}")
        
        self.model = model
        self.model.to(self.device)

        # 确定评价指标名称
        if self.model.type == "classifier":
            self.metric_name = "F1Score"
        else:
            self.metric_name = "map" # Mean Average Precision
            
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 初始化状态记录
        self.current_state = {
            "loss": 0,
            "metric": {self.metric_name: 0},
            "epoch": 0,
        }
        self.best_state = {
            "loss": 0,
            "metric": {self.metric_name: 0},
            "epoch": 0,
        }

        self.stop = False
        self.max_epoch = self.config.epochs
        self.epochs_run = 0
        self.n_gpu = n_gpu
        
        # 移动评估指标计算器到设备
        try:
            self.metric_calculator = metric_calculator.to(self.device)
        except Exception as e:
            print(f"⚠️  评估指标计算器移动设备失败: {e}")
            print("⚠️  评估指标计算器将保持在原设备")
            self.metric_calculator = metric_calculator

        # 初始化日志记录器和文件夹
        if not os.path.exists(self.config.work_dir):
            os.mkdir(self.config.work_dir)
        
        # 初始化 TensorBoard Writer (自动分流逻辑)
        log_path = f"{self.config.work_dir}/{self.config.experiment_name}/logs"
        if log_subdir:
            log_path = os.path.join(log_path, log_subdir)
            
        self.summary_writer = SummaryWriter(log_dir=log_path)
        self.summary_writer.add_text("model/name", self.config.model.name)

        # 如果有 Checkpoint，加载断点
        if self.config.model.checkpoint is not None:
            self._load_snapshot(self.config.model.checkpoint)

    def _save_snapshot(self):
        """
        保存模型权重和训练状态
        """
        metric_score = self.best_state["metric"][self.metric_name]
        
        # 获取模型状态
        state = self.model.state_dict()
            
        snapshot = {
            "MODEL_STATE": state,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict() if self.scheduler else None,
            "EPOCHS_RUN": self.best_state["epoch"],
            "Loss": self.best_state["loss"],
            "Metric": self.best_state["metric"],
        }
        save_path = os.path.join(self.config.work_dir, self.config.experiment_name)
        # 文件名包含 epoch, metric 分数和 loss，方便查看    1.0这里改了一下，windows不能使用：，换成了_
        save_name = f"{self.config.model.name}_epoch-{self.best_state['epoch']}_{self.metric_name}-{metric_score:.2}_loss-{self.best_state['loss']:.2}.pth"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(snapshot, os.path.join(save_path, save_name))
        print(f"Save model {self.config.model.name} || {self.metric_name}:{metric_score:.2}")

    def _load_snapshot(self, snapshot_path):
        """
        加载断点继续训练
        """
        snapshot = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        if self.scheduler and "SCHEDULER_STATE" in snapshot and snapshot["SCHEDULER_STATE"]:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        # 恢复最佳状态记录，防止刚开始训练就覆盖了之前的最佳模型
        self.best_state["epoch"] = snapshot["EPOCHS_RUN"]
        self.best_state["loss"] = snapshot["Loss"]
        self.best_state["metric"] = snapshot["Metric"]
        print(f"Loaded model from {snapshot_path}")

    def test(self):
        """
        在测试集上运行推理
        """
        self.model.eval() # 切换到评估模式
        if self.test_data is None:
            raise Exception("Cannot test without test data")

        # Logger 是自定义的上下文管理器，用于打印漂亮的进度条
        with Logger("Test", self.max_epoch, len(self.test_data), self.config.log_every, self.device) as logger:
            for iteration, (images, targets) in enumerate(self.test_data):
                # 将数据移动到设备
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with torch.no_grad(): # 禁用梯度计算
                    output = self.model(images)

                # 计算指标
                metric = self.metric_calculator(output, targets)

                logger.log_iteration(iteration + 1, self.current_state["epoch"], metrics=metric)

            # 记录到 TensorBoard
            for key, value in metric.items():
                self.summary_writer.add_scalar(f"{key}/Test", value, self.current_state["epoch"])

    def val(self):
        """
        在验证集上运行推理并保存最佳模型
        """
        self.model.eval()
        if self.val_data is None:
            raise Exception("Cannot validate without validation data")
        with Logger("Eval", self.max_epoch, len(self.val_data), self.config.log_every, self.device) as logger:
            for iteration, (images, targets) in enumerate(self.val_data):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    output = self.model(images)

                metric = self.metric_calculator(output, targets)
                logger.log_iteration(iteration + 1, self.current_state["epoch"], metrics=metric)

            # 统计和保存模型
            self.current_state["metric"] = logger.metric_averager.value

            for key, value in self.current_state["metric"].items():
                self.summary_writer.add_scalar(f"{key}/Eval", value, self.current_state["epoch"])

            # --- 核心逻辑：保存最佳模型 ---
            # 如果当前指标比历史最佳还要好 (超过了阈值)，则更新最佳状态并保存
            if (
                self.current_state["metric"][self.metric_name] - self.best_state["metric"][self.metric_name]
            ) > self.config.early_stopping.metric:
                self.best_state["metric"] = self.current_state["metric"]
                self.best_state["loss"] = self.current_state["loss"]
                self.best_state["epoch"] = self.current_state["epoch"]

                self._save_snapshot()

    def early_stop(self):
        """
        检查是否满足早停条件 (耐心值耗尽且指标未提升)
        """
        if (
            self.current_state["epoch"] - self.best_state["epoch"] >= self.config.early_stopping.epochs
            and self.current_state["metric"][self.metric_name] - self.best_state["metric"][self.metric_name]
            <= self.config.early_stopping.metric
        ):
            return True
        else:
            return False

    def train(self):
        """
        主训练循环
        """
        if self.train_data is None:
            raise Exception("Cannot train without training data")
        
        # 循环 Epoch
        for epoch in range(self.epochs_run, self.max_epoch):
            # 1. 检查是否需要早停
            if self.early_stop():
                self.stop = True
                
            if self.stop:
                break

            # 2. 设置训练模式
            self.model.train()
            self.current_state["epoch"] = epoch
                
            with Logger("Train", self.max_epoch, len(self.train_data), self.config.log_every, self.device) as logger:
                # 循环 Batch
                for iteration, (images, targets) in enumerate(self.train_data):
                    self.optimizer.zero_grad() # 清空梯度

                    # 移动数据到设备
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # 前向传播计算 Loss
                    loss = self.model(images, targets)

                    # 反向传播
                    loss.backward()
                    self.optimizer.step()

                    logger.log_iteration(iteration + 1, self.current_state["epoch"], loss.item())

                # 更新学习率
                if self.scheduler is not None:
                    if hasattr(self.config.scheduler, 'name') and self.config.scheduler.name == "ReduceLROnPlateau":
                        self.scheduler.step(self.current_state["loss"])
                    else:
                        self.scheduler.step()

                # 记录 Loss 到 TensorBoard
                self.current_state["loss"] = logger.loss_averager.value
                self.summary_writer.add_scalar(
                    "loss/Train", self.current_state["loss"], self.current_state["epoch"]
                )

            # 定期验证
            if self.config.eval_every > 0 and self.current_state["epoch"] % self.config.eval_every == 0:
                self.val()

            # 定期测试
            if self.config.test_every > 0 and self.current_state["epoch"] % self.config.test_every == 0:
                self.test()