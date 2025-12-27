from functools import partial

# 自己写的模型
from .classifiers.se_resnet import se_resnet18

# Wrapper (之前改好的 base_model_my)
from .classifiers.base_model_my import ClassifierModel
from .model import HaGRIDModel

# 上面的 import 语句已经把 "ClassifierModel" 这个名字
# 偷梁换柱成了base_model_my.py 里的那个新类
classifiers_list = {
    # 注册 SE-ResNet18
    # 我们用 ClassifierModel 包装它，但 model 参数传我们写的函数
    "SE_ResNet18": partial(ClassifierModel, model=se_resnet18),
}