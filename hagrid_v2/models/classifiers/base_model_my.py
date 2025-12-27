from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from models.model import HaGRIDModel


class ClassifierModel(HaGRIDModel):
    def __init__(self, model: nn.Module, **kwargs):
        """
        Parameters
        ----------
        model: nn.Module
            The model constructor (e.g., torchvision.models.resnet18)
        kwargs:
            Includes 'num_classes', 'pretrained', etc.
        """
        super().__init__()
        
        # 1. 提取关键参数
        # 从 kwargs 中把 num_classes 拿出来，防止直接传给预训练模型报错
        num_classes = kwargs.pop("num_classes", None)
        pretrained = kwargs.pop("pretrained", False)
        
        # 2. 初始化模型逻辑
        if pretrained:
            # === 方案 A: 预训练模式 ===
            # 先不传 num_classes，让它默认加载 ImageNet 的 1000 类结构和权重
            # (torchvision 新版使用 weights参数，旧版使用 pretrained)
            try:
                self.hagrid_model = model(weights="DEFAULT", **kwargs)
            except TypeError:
                # 兼容旧版 torchvision
                self.hagrid_model = model(pretrained=True, **kwargs)
            
            # === 换头手术 (核心修改) ===
            # 加载完 1000 类权重后，把最后一层替换成我们的 num_classes (如 8)
            self._replace_head(num_classes)
            
        else:
            # === 方案 B: 从头训练模式 ===
            # 直接传入 num_classes，从零初始化
            if num_classes is not None:
                self.hagrid_model = model(num_classes=num_classes, **kwargs)
            else:
                self.hagrid_model = model(**kwargs)

        self.criterion = None

    def _replace_head(self, num_classes):
        """
        辅助函数：自动识别模型类型并替换最后一层
        """
        if num_classes is None:
            return

        # 1. 针对 ResNet 系列 (最后一层叫 fc)
        if hasattr(self.hagrid_model, "fc"):
            in_features = self.hagrid_model.fc.in_features
            
            # 改进后的写法 (MLP Head):
            # 结构：Linear -> BN -> ReLU -> Dropout -> Linear
            self.hagrid_model.fc = nn.Sequential(
                nn.Linear(in_features, 512),        # 降维到 512
                nn.BatchNorm1d(512),                # 归一化，加速收敛
                nn.ReLU(inplace=True),              # 激活函数，增加非线性
                nn.Dropout(p=0.5),                  # 防止过拟合 (关键!)
                nn.Linear(512, num_classes)         # 最终输出 7 类
            )
        
        # 2. 针对 MobileNet / VGG / EfficientNet 系列 (最后一层叫 classifier)
        elif hasattr(self.hagrid_model, "classifier"):
            # 如果 classifier 是一个序列 (Sequential)
            if isinstance(self.hagrid_model.classifier, nn.Sequential):
                # 找到最后一层线性层
                last_layer_idx = len(self.hagrid_model.classifier) - 1
                for i in reversed(range(len(self.hagrid_model.classifier))):
                    if isinstance(self.hagrid_model.classifier[i], nn.Linear):
                        last_layer_idx = i
                        break
                
                # 替换它
                in_features = self.hagrid_model.classifier[last_layer_idx].in_features
                self.hagrid_model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)
            else:
                # 如果 classifier 只是单个层
                in_features = self.hagrid_model.classifier.in_features
                self.hagrid_model.classifier = nn.Linear(in_features, num_classes)
        
        # 3. 针对 ViT 系列 (最后一层叫 heads)
        elif hasattr(self.hagrid_model, "heads"):
             if hasattr(self.hagrid_model.heads, "head"):
                in_features = self.hagrid_model.heads.head.in_features
                self.hagrid_model.heads.head = nn.Linear(in_features, num_classes)

    def __call__(self, images: list[Tensor], targets: Dict = None) -> Dict:
        """
        前向传播逻辑
        """
        # 将图片列表堆叠成一个 Batch Tensor
        image_tensors = torch.stack(images)
        
        # 模型推理
        model_output = self.hagrid_model(image_tensors)
        
        # 包装输出
        output_dict = {"labels": model_output}
        
        # 如果有标签，计算 Loss
        if targets is None:
            return output_dict
        else:
            target_tensors = torch.stack([target["labels"] for target in targets])
            # self.criterion 在 utils.py 里被赋值为 CrossEntropyLoss
            return self.criterion(output_dict["labels"], target_tensors)

    # 原代码中的 def criterion(...) 其实是个占位符或有逻辑问题
    # 因为 utils.py 会直接覆盖 self.criterion 属性。
    # 这里我们不需要显式定义它，留着 self.criterion = None 即可。