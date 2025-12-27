from functools import partial
from torchvision import models

# 自己写的模型
from .classifiers.se_resnet import se_resnet18
from .classifiers.cbam_resnet import cbam_resnet18

# Wrapper (之前改好的 base_model_my)
from .classifiers.base_model_my import ClassifierModel
from .detectors import SSDLiteMobilenet_large
from .model import HaGRIDModel

detectors_list = {
    "SSDLiteMobileNetV3Large": SSDLiteMobilenet_large,
}

# 下面这个字典完全不用动
# 因为上面的 import 语句已经把 "ClassifierModel" 这个名字
# 偷梁换柱成了base_model_my.py 里的那个新类
classifiers_list = {
    # 注册 SE-ResNet18
    # 这里用 ClassifierModel 包装它，但 model 参数传自己写的函数
    "SE_ResNet18": partial(ClassifierModel, model=se_resnet18),
    "CBAM_ResNet18": partial(ClassifierModel, model=cbam_resnet18),
    # 其他模型
    "ResNet18": partial(ClassifierModel, models.resnet18),
    "ResNet152": partial(ClassifierModel, models.resnet152),
    "MobileNetV3_small": partial(ClassifierModel, models.mobilenet_v3_small),
    "MobileNetV3_large": partial(ClassifierModel, models.mobilenet_v3_large),
    "ConvNeXt_base": partial(ClassifierModel, models.convnext_base),
}