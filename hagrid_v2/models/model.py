from typing import Dict, Iterator, List, Tuple
import torch
from torch import Tensor, nn


class HaGRIDModel:
    """
    Torchvision class wrapper
    """

    def __init__(self):
        self.hagrid_model = None
        self.type = None
        self.device = None  # 添加设备跟踪

    def __call__(self, img: Tensor, targets: Dict = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        raise NotImplementedError

    def to(self, device: str):
        """
        将模型移动到指定设备，支持自动降级到 CPU
        """
        # 如果请求的是 CUDA 但不可用，自动降级到 CPU
        if "cuda" in str(device) and not torch.cuda.is_available():
            device = "cpu"
            print(f"⚠️  CUDA 不可用，自动切换到 {device}")
        
        self.device = device
        self.hagrid_model = self.hagrid_model.to(device)
        
        # 可选：打印设备信息
        if hasattr(self.hagrid_model, 'device'):
            print(f"✅ 模型已移动到设备: {device}")
        
        return self  # 保持链式调用兼容性

    def get_device(self):
        """获取当前模型所在的设备"""
        return self.device if self.device else "cpu"

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.hagrid_model.parameters()

    def train(self):
        self.hagrid_model.train()

    def eval(self):
        self.hagrid_model.eval()

    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        self.hagrid_model.load_state_dict(state_dict)

    def state_dict(self):
        return self.hagrid_model.state_dict()