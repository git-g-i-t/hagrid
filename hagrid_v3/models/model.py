import mindspore as ms
from mindspore import nn

class HaGRIDModel:
    """
    简化后的 MindSpore 模型包装基类
    """
    def __init__(self):
        self.hagrid_model = None # 存储实际的 nn.Cell
        self.device = "CPU"

    def train(self):
        """设置模型为训练模式"""
        self.hagrid_model.set_train(True)

    def eval(self):
        """设置模型为评估模式"""
        self.hagrid_model.set_train(False)

    def parameters(self):
        """返回参数迭代器，用于优化器"""
        return self.hagrid_model.get_parameters()

    def save_ckpt(self, path):
        """保存 MindSpore 权重"""
        ms.save_checkpoint(self.hagrid_model, path)

    def load_ckpt(self, path):
        """加载 MindSpore 权重"""
        ms.load_param_into_net(self.hagrid_model, ms.load_checkpoint(path))