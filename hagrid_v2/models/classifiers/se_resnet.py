import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3, conv1x1

# 1. 定义注意力模块 (SE-Block)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 两个全连接层，先降维再升维，学习通道权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 输出 0~1 之间的权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 将权重乘回原始特征图
        return x * y.expand_as(x)

# 2. 定义带有注意力的残差块 (SE-BasicBlock)
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 标准 ResNet 卷积部分
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        # 插入 SE 注意力模块 
        self.se = SELayer(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 在残差连接之前，先过注意力模块
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 3. 定义主网络结构 (SE-ResNet18)
# 3. SEResNet (支持动态层数，兼容搜索脚本)
class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, hidden_layers=None): 
        super(SEResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        # --- 骨干网络 ---
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        
        # 如果 hidden_layers 为空，这个循环不执行，直接接最后一层 Linear
        # 这就等同于原始的单层结构
        for hidden_dim in hidden_layers:
            layers_list.append(nn.Linear(current_dim, hidden_dim))
            layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(nn.ReLU(inplace=True))
            layers_list.append(nn.Dropout(p=0.5))
            current_dim = hidden_dim
            
        # 最终输出层
        layers_list.append(nn.Linear(current_dim, num_classes))
        
        self.fc = nn.Sequential(*layers_list)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- 工厂函数 ---

# 只保留这一个，默认就是单层
def se_resnet18(num_classes=7, **kwargs):
    # kwargs 里可能会包含 hidden_layers (如果搜索脚本传进来的话)
    # 如果没传，SEResNet 默认 hidden_layers=None，即单层
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)