from functools import partial
from torchvision import models

# 1. 导入你刚才写的 SimpleCNN 类
# 注意路径：.classifiers.simple_cnn 是文件名，SimpleCNN 是类名
from .classifiers.simple_cnn import SimpleCNN  

# 你的 Wrapper (之前改好的 base_model_my)
from .classifiers.base_model_my import ClassifierModel
from .detectors import SSDLiteMobilenet_large
from .model import HaGRIDModel

detectors_list = {
    "SSDLiteMobileNetV3Large": SSDLiteMobilenet_large,
}

# 下面这个字典完全不用动
# 因为上面的 import 语句已经把 "ClassifierModel" 这个名字
# 偷梁换柱成了你 base_model_my.py 里的那个新类
classifiers_list = {
    "SimpleCNN": partial(ClassifierModel, model=SimpleCNN),
    # 其他模型
    "ResNet18": partial(ClassifierModel, models.resnet18),
    "ResNet152": partial(ClassifierModel, models.resnet152),
    "MobileNetV3_small": partial(ClassifierModel, models.mobilenet_v3_small),
    "MobileNetV3_large": partial(ClassifierModel, models.mobilenet_v3_large),
    "ConvNeXt_base": partial(ClassifierModel, models.convnext_base),
}