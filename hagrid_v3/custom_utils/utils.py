import random
from collections import defaultdict
from time import gmtime, strftime
from typing import Dict
from sklearn.metrics import f1_score # 引入 sklearn 的标准计算函数
import albumentations as A
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
# 移除 ToTensorV2，MindSpore 推荐在数据管道中处理
from omegaconf import DictConfig
from omegaconf import OmegaConf 
import pandas as pd
from sklearn.metrics import confusion_matrix
# 导入类别名映射
from constants import targets as target_dict

# 导入自定义的模型列表
from models import classifiers_list

def get_available_device():
    """
    获取可用设备 (MindSpore CPU 版固定返回 CPU)
    """
    return "CPU"

class F1ScoreWithLogging:
    def __init__(self, task, num_classes):
        self.num_classes = num_classes
        self.all_preds = []
        self.all_labels = []
        self.device = "CPU"

    def clear_accumulated(self):
        """测试/验证开始前清空"""
        self.all_preds = []
        self.all_labels = []

    def __call__(self, preds, targets):
        """
        每个 Batch 调用：存数据，返回一个临时准确率供进度条显示
        """
        target_list = [t["labels"] for t in targets]
        target = ops.stack(target_list).asnumpy().flatten()
        pred_logits = preds["labels"]
        y_pred = ops.argmax(pred_logits, dim=1).asnumpy().flatten()
        
        # 累积全量数据
        self.all_preds.extend(y_pred)
        self.all_labels.extend(target)

        # 进度条上显示 Accuracy 比显示错误的 Batch-F1 更有意义
        batch_acc = np.mean(y_pred == target)
        return {"Accuracy": float(batch_acc)}

    def print_final_matrix(self, epoch, stage="Eval"):
        """
        整个数据集跑完后调用：计算真正的全量 F1 并打印矩阵
        """
        if len(self.all_labels) == 0: return {}

        from constants import targets as target_dict
        class_names = [target_dict[i] for i in range(self.num_classes)]
        
        # 1. 计算混淆矩阵
        cm = confusion_matrix(self.all_labels, self.all_preds, labels=list(range(self.num_classes)))
        df = pd.DataFrame(cm, index=[f"真:{n}" for n in class_names], 
                             columns=[f"预:{n}" for n in class_names])
        
        # 2. 计算真正的全量 Macro-F1
        real_f1 = f1_score(self.all_labels, self.all_preds, average='macro')
        
        print(f"\n" + "="*60)
        print(f"[{stage} 总结] 样本总数: {len(self.all_labels)}")
        print(f"最终 F1-Score (Macro): {real_f1:.4f}") 
        print("="*60)
        print(df)
        print("="*60 + "\n")
        
        return {"F1Score": real_f1}

class Logger:
    """
    自定义日志记录器 (适配 MindSpore)
    """
    def __init__(self, train_state: str, max_epochs: int, dataloader_len: int, log_every: int, device: str = "CPU"):
        self.dataloader_len = dataloader_len
        self.max_epochs = max_epochs
        self.train_state = train_state
        self.log_every = log_every
        self.device = device
        self.loss_averager = LossAverager()
        self.metric_averager = MetricAverager()

    def log_iteration(self, iteration: int, epoch: int, loss: float = None, metrics: dict = None):
        if (iteration % self.log_every == 0) or (iteration == self.dataloader_len):
            log_str = f"Time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())} "
            log_str += f"{self.train_state} ---- Epoch [{epoch}/{self.max_epochs}], Iteration [{iteration}/{self.dataloader_len}]:"
            
            if self.train_state == "Train" and loss is not None:
                self.loss_averager.update(loss)
                log_str += f" Loss: {self.loss_averager.value:.4f}"
            
            if self.train_state in ["Eval", "Test"] and metrics is not None:
                self.metric_averager.update(metrics)
                if iteration == self.dataloader_len:
                    for metric_name, metric_value in self.metric_averager.value.items():
                        log_str += f" {metric_name}: {metric_value:.4f}"
            print(log_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MetricAverager:
    """
    指标平均值计算器 (适配 MindSpore Tensor)
    """
    def __init__(self):
        self.current_total = defaultdict(float)
        self.iterations = 0

    def update(self, values: Dict):
        for key, value in values.items():
            # MindSpore Tensor 转 float 使用 .asnumpy().item() 或直接 float()
            val = float(value) if isinstance(value, (Tensor, np.ndarray)) else value
            self.current_total[key] += val
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return {}
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
        # 兼容 MindSpore Tensor 和普通 float
        val = float(value) if hasattr(value, "asnumpy") else value
        self.current_total += val
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        return self.current_total / self.iterations


def get_transform(transform_config: DictConfig, model_type: str):
    """
    构建数据增强 Pipeline (适配 MindSpore)
    去掉了 ToTensorV2，因为数据集 __getitem__ 需要返回 Numpy
    """
    transforms_list = []
    
    for key, params in transform_config.items():
        real_params = OmegaConf.to_container(params, resolve=True)
        transforms_list.append(getattr(A, key)(**real_params))

    # 移除 ToTensorV2()，保持输出为 Numpy 数组格式
    # transforms_list.append(ToTensorV2())

    if model_type == "detector":
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["class_labels"]),
        )
    elif model_type == "classifier":
        return A.Compose(transforms_list)


def build_model(config: DictConfig):
    from models import classifiers_list
    import mindspore.nn as nn
    model_name = config.model.name
    # 只需要传入 num_classes，不需要再手动写 ClassifierModel(se_resnet18, ...)
    model_config = {"num_classes": len(config.dataset.targets)}
   
    if model_name in classifiers_list:
        # 这里的调用会触发 partial 执行：
        # 实际效果等于 ClassifierModel(model=se_resnet18, num_classes=X)
        model = classifiers_list[model_name](**model_config)
        
        # 绑定 MindSpore 损失函数
        model.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model.type = "classifier"
    else:
        raise Exception(f"Unknown model {model_name}")

    return model


def set_random_seed(seed: int = 42, deterministic: bool = False) -> int:
    """
    设置随机种子 (适配 MindSpore)
    """
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed) # MindSpore 全局种子
    return seed