import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3, conv1x1

# ================= CBAM 核心模块 =================

class ChannelAttention(nn.Module):
    """
    CBAM 的通道注意力部分 (Channel Attention Module)
    注意：CBAM 原始论文中不仅使用了 GlobalAvgPool，还并行使用了 GlobalMaxPool
    然后共享同一个 MLP，最后将两个结果相加。
    这比纯 SE (只有 AvgPool) 更强大。
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP (Shared MLP)
        # 1x1 卷积可以替代全连接层，且不用 flatten，方便处理
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. AvgPool 分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 2. MaxPool 分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 3. 相加并激活
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    CBAM 的空间注意力部分 (Spatial Attention Module)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 将 AvgPool 和 MaxPool 的结果拼接后，通过一个卷积层压缩为 1 个通道
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿通道轴做平均池化 -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 2. 沿通道轴做最大池化 -> (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 3. 拼接 -> (B, 2, H, W)
        x = torch.cat([avg_out, max_out], dim=1)
        # 4. 卷积 + 激活
        x = self.conv1(x)
        return self.sigmoid(x)

# ================= 集成到 ResNet Block =================

class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CBAMBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 标准 ResNet 卷积部分
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # 🔥 插入 CBAM 模块 🔥
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 🔥 CBAM 推理逻辑：先通道，后空间 🔥
        # 1. Channel Attention
        out = self.ca(out) * out
        # 2. Spatial Attention
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ================= CBAM-ResNet 主体结构 =================

class CBAMResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(CBAMResNet, self).__init__()
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
        
        # 四个 Stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 简单分类头 (保持和 ResNet18 一致，为了控制变量)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

def cbam_resnet18(num_classes=7, **kwargs):
    """
    构建 CBAM-ResNet18 模型
    """
    return CBAMResNet(CBAMBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
