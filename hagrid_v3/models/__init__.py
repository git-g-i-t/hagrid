from functools import partial
#from torchvision import models

# 自己写的模型
from .classifiers.se_resnet import se_resnet18
#from .classifiers.cbam_resnet import cbam_resnet18

# 你的 Wrapper (之前改好的 base_model_my)
from .classifiers.base_model_my import ClassifierModel
from .model import HaGRIDModel

# 下面这个字典完全不用动
# 因为上面的 import 语句已经把 "ClassifierModel" 这个名字
# 偷梁换柱成了你 base_model_my.py 里的那个新类
classifiers_list = {
    # 注册 SE-ResNet18
    # 注意：这里我们用 ClassifierModel 包装它，但 model 参数传我们写的函数
    "SE_ResNet18": partial(ClassifierModel, model=se_resnet18),
    #"CBAM_ResNet18": partial(ClassifierModel, model=cbam_resnet18),
    # 其他模型
    #"ResNet18": partial(ClassifierModel, models.resnet18),
}