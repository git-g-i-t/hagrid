import mindspore as ms
from mindspore import nn, ops, Tensor
from typing import Dict, List
from models.model import HaGRIDModel

class ClassifierModel(HaGRIDModel):
    # 参数名为 model，与 partial 风格对齐
    def __init__(self, model, num_classes=7, **kwargs):
        super().__init__()
        
        # 使用传入的构造函数初始化底层网络
        self.hagrid_model = model(num_classes=num_classes, **kwargs)
        self.criterion = None
        self.type = "classifier"

    def __call__(self, images, targets=None) -> Dict:
        # 保持维度检查逻辑
        if isinstance(images, (list, tuple)):
            image_tensors = ops.stack(images)
        elif isinstance(images, ms.Tensor) and images.ndim == 4:
            image_tensors = images
        else:
            image_tensors = images

        model_output = self.hagrid_model(image_tensors)
        output_dict = {"labels": model_output}
        
        if targets is not None:
            if isinstance(targets, (list, tuple)):
                target_tensors = ops.stack([t["labels"] for t in targets])
            else:
                target_tensors = targets
            if self.criterion is not None:
                return self.criterion(output_dict["labels"], target_tensors)
        return output_dict