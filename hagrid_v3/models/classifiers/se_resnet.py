import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Normal, Constant, HeNormal

# 辅助函数：卷积层定义（对齐 PyTorch 的默认行为）
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=1, has_bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     pad_mode='pad', padding=0, has_bias=False)

# 1. 定义注意力模块 (SE-Block)
class SELayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 两个全连接层
        # MindSpore 中 Linear 对应 nn.Dense
        self.fc = nn.SequentialCell([
            nn.Dense(channel, channel // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel, has_bias=False),
            nn.Sigmoid()
        ])

    def construct(self, x):
        b, c, _, _ = x.shape
        # view(b, c) 对应 reshape(b, c)
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        # 权重乘回原始特征图，MindSpore 自动处理广播机制
        return x * y

# 2. 定义带有注意力的残差块 (SE-BasicBlock)
class SEBasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        # SE 注意力模块
        self.se = SELayer(planes)
        
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用注意力
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 3. 定义主网络结构 (SEResNet)
class SEResNet(nn.Cell):
    def __init__(self, block, layers, num_classes=7, hidden_layers=None): 
        super(SEResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        
        # --- 骨干网络 ---
        # padding=3 且 pad_mode='pad' 才能对齐 PyTorch
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, 
                               pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 动态分类头 ---
        input_features = 512 * block.expansion
        if hidden_layers is None:
            hidden_layers = [] 
            
        layers_list = []
        current_dim = input_features
        
        for hidden_dim in hidden_layers:
            layers_list.append(nn.Dense(current_dim, hidden_dim))
            layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(p=0.5))
            current_dim = hidden_dim
            
        layers_list.append(nn.Dense(current_dim, num_classes))
        self.fc = nn.SequentialCell(layers_list)

        # 权重初始化
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            ])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(layers)

    def _init_weights(self):
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                # 必须显式指定 mode='fan_out' 以对齐 PyTorch ResNet
                m.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'), 
                    m.weight.shape))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(ms.common.initializer.initializer(ms.common.initializer.Constant(1), m.gamma.shape))
                m.beta.set_data(ms.common.initializer.initializer(ms.common.initializer.Constant(0), m.beta.shape))
            elif isinstance(m, nn.Dense):
                # 初始化的关键：给分类头一个较小的随机分布，防止预测值直接冲向某个类
                m.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.Normal(sigma=0.01), m.weight.shape))
                if m.has_bias:
                    m.bias.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Constant(0), m.bias.shape))
                    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # torch.flatten(x, 1) 对应 ops.flatten
        x = ops.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

def se_resnet18(num_classes=7, **kwargs):
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)