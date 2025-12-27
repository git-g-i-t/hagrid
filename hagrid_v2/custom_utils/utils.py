# 这个文件是项目的通用工具箱
# 包含了评价指标计算、日志记录、图像增强构建、模型构建工厂以及随机种子设置等基础功能。
# 它是连接配置（Config）、模型（Model）和训练流程（Trainer）的纽带。
import random
from collections import defaultdict
from time import gmtime, strftime
from typing import Dict

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torchmetrics import F1Score
from omegaconf import OmegaConf 
# 导入自定义的模型列表（在 models/__init__.py 中定义）
from models import classifiers_list, detectors_list

TORCH_VERSION = torch.__version__


def get_available_device():
    """
    获取可用设备，优先使用 GPU，如果没有则使用 CPU
    """
    if torch.cuda.is_available():
        return "cuda:0"  # 使用第一个 GPU
    else:
        return "cpu"


class F1ScoreWithLogging:
    """
    对 torchmetrics.F1Score 的封装类。
    主要作用是适配 Trainer 的接口，处理输入输出格式，并支持移动到 GPU。
    """
    def __init__(self, task, num_classes):
        """
        Parameters
        ----------
        task : str
            任务类型 ('binary' 或 'multiclass')
        num_classes : int
            类别数量
        """
        # 初始化 F1Score 计算器
        self.f1_score = F1Score(task=task, num_classes=num_classes)
        self.device = get_available_device()

    def to(self, device):
        """
        将指标计算器移动到指定的设备 (CPU/GPU)
        支持自动降级到 CPU
        """
        # 如果请求的是 CUDA 但不可用，自动降级到 CPU
        if "cuda" in str(device) and not torch.cuda.is_available():
            device = "cpu"
            print(f"评估指标计算器: CUDA 不可用，自动切换到 {device}")
        
        try:
            self.f1_score = self.f1_score.to(device)
            self.device = device
        except Exception as e:
            print(f"评估指标计算器移动设备失败: {e}")
            print("评估指标计算器将保持在原设备")
        
        return self

    def __call__(self, preds, targets):
        """
        计算 F1 分数
        
        Parameters
        ----------
        preds : dict
            模型输出的预测结果，通常是一个字典，包含 "labels" (logits/probs)
        targets : list
            真实标签列表，每个元素是一个字典 (包含 "labels")
        """
        # 将 target 列表堆叠成 Tensor
        target = torch.stack([target["labels"] for target in targets])
        
        # 获取预测的类别索引
        pred_labels = preds["labels"].to(self.device).argmax(1)
        
        # =========== 🕵️‍♂️ 调试代码开始  ===========
        # 打印前10个预测结果和真实标签，看看它到底在猜什么
        print(f"\n[DEBUG] 预测: {pred_labels[:10].tolist()}")
        print(f"[DEBUG] 真实: {target[:10].tolist()}")
        # =========== 🕵️‍♂️ 调试代码结束 ===========================


        # preds["labels"].argmax(1): 获取概率最大的类别索引
        # 计算预测值与真实值的 F1 分数
        result = self.f1_score(preds["labels"].argmax(1), target)
        
        # 返回字典格式，方便 Logger 记录
        return {"F1Score": result}


class Logger:
    """
    自定义日志记录器 (Context Manager)
    功能：
    1. 格式化打印训练进度、时间、Loss 和 Metrics。
    2. 维护 Loss 和 Metrics 的滑动平均值 (Averager)。
    3. 只在主设备上打印，避免多卡训练时刷屏。
    """
    def __init__(self, train_state: str, max_epochs: int, dataloader_len: int, log_every: int, device: str = "cpu"):
        """
        Parameters
        ----------
        train_state : str
            当前状态: "Train", "Eval" 或 "Test"
        max_epochs : int
            总 Epoch 数
        dataloader_len : int
            当前 DataLoader 的长度 (Batch 总数)
        log_every : int
            每隔多少个 iteration 打印一次日志
        device : str
            当前使用的设备
        """
        self.dataloader_len = dataloader_len
        self.max_epochs = max_epochs
        self.train_state = train_state
        self.log_every = log_every
        self.device = device
        # 初始化平均值计算器
        self.loss_averager = LossAverager()
        self.metric_averager = MetricAverager()

    def log_iteration(self, iteration: int, epoch: int, loss: float = None, metrics: dict = None):
        """
        记录当前迭代的信息
        """
        # 只有在指定的间隔 (log_every) 或最后一个 batch 时才打印
        if (iteration % self.log_every == 0) or (iteration == self.dataloader_len):
            # 获取当前时间
            log_str = f"Time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())} "
            log_str += f"{self.train_state} ---- Epoch [{epoch}/{self.max_epochs}], Iteration [{iteration}/{self.dataloader_len}]:"
            
            # 如果是训练阶段，记录 Loss
            if self.train_state == "Train" and loss is not None:
                self.loss_averager.update(loss)
                log_str += f" Loss: {self.loss_averager.value}"
            
            # 如果是验证/测试阶段，记录 Metrics
            if self.train_state in ["Eval", "Test"] and metrics is not None:
                # 清理掉不需要打印的 key (如果有的话)
                try:
                    del metrics["classes"]
                except KeyError:
                    pass
                
                self.metric_averager.update(metrics)
                
                # 只有在跑完整个验证集后 (最后一个 iteration)，才打印最终的平均指标
                if iteration == self.dataloader_len:
                    for metric_name, metric_value in self.metric_averager.value.items():
                        log_str += f" {metric_name}: {metric_value}"
            print(log_str)

    # 上下文管理器协议：支持 `with Logger(...) as logger:` 语法
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MetricAverager:
    """
    指标平均值计算器
    用于在验证过程中累加每个 Batch 的指标，最后求平均。
    """
    def __init__(self):
        self.current_total = defaultdict(float) # 使用 defaultdict 防止 key 不存在报错
        self.iterations = 0

    def update(self, values: Dict):
        for key, value in values.items():
            self.current_total[key] += value.item()
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            # 计算平均值
            metrics = {key: value / self.iterations for key, value in self.current_total.items()}
            return metrics


class LossAverager:
    """
    Loss 平均值计算器
    """
    def __init__(self):
        self.iterations = 0
        self.current_total = 0

    def update(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return self.current_total / self.iterations


def get_transform(transform_config: DictConfig, model_type: str):
    """
    构建数据增强 Pipeline
    修复了 ListConfig 类型导致的 TypeError
    """
    transforms_list = []
    
    for key, params in transform_config.items():
        # OmegaConf 读取的参数是 DictConfig/ListConfig 类型
        # Albumentations 不认这些类型，必须转回 Python 原生的 dict/list
        real_params = OmegaConf.to_container(params, resolve=True)
        
        # 实例化增强方法
        transforms_list.append(getattr(A, key)(**real_params))

    transforms_list.append(ToTensorV2())

    if model_type == "detector":
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["class_labels"]),
        )
    elif model_type == "classifier":
        return A.Compose(transforms_list)


def build_model(config: DictConfig):
    """
    模型构建工厂函数
    根据配置文件中的 model.name 实例化对应的模型。
    """
    model_name = config.model.name
   
    # 这样无论以后是用 7 类、18 类还是 34 类，代码都能自动适应，不用再改了
    model_config = {"num_classes": len(config.dataset.targets), "pretrained": config.model.pretrained}
   
    # 情况 1: 目标检测模型 (如 SSDLite)
    if model_name in detectors_list:
        # 检测任务通常需要一个额外的 "背景" 类，所以 +1 (变成 35 类)
        model_config["num_classes"] += 1
        # 更新检测模型特有的配置 (输入尺寸、均值方差用于 Backbone 预处理)
        model_config.update(
            {
                "pretrained_backbone": config.model.pretrained_backbone,
                "img_size": config.dataset.img_size,
                "img_mean": config.dataset.img_mean,
                "img_std": config.dataset.img_std,
            }
        )
        # 实例化检测模型
        model = detectors_list[model_name](**model_config)
        # 打上标记，后续 load_train_objects 会根据这个标记加载 DetectionDataset
        model.type = "detector"
        
    # 情况 2: 图像分类模型 (如 ResNet, MobileNet)
    elif model_name in classifiers_list:
        # 实例化分类模型
        model = classifiers_list[model_name](**model_config)
        # 绑定损失函数 (如 CrossEntropyLoss)
        model.criterion = getattr(torch.nn, config.criterion)()
        # 打上标记，后续会加载 ClassificationDataset
        model.type = "classifier"
    else:
        raise Exception(f"Unknown model {model_name}")

    return model


def set_random_seed(seed: int = 42, deterministic: bool = False) -> int:
    """
    设置随机种子，保证实验可复现。

    Args:
        seed (int, optional): 种子值.
        deterministic (bool): 是否强制使用确定性算法 (会降低训练速度但保证结果完全一致).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        # 为所有 GPU 设置种子
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print(
                "torch.backends.cudnn.benchmark is going to be set as "
                "`False` to cause cuDNN to deterministically select an "
                "algorithm"
            )
        # 禁用 cudnn benchmark (自动寻找最快算法)，因为它有随机性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if TORCH_VERSION >= "1.10.0":
            torch.use_deterministic_algorithms(True)
    return seed