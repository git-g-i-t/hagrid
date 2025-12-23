import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    [升级版] SimpleCNN
    虽然名字没变，但结构已经升级为 VGG 风格的深层网络。
    特点：更深 (4个Block)，更宽 (最高512通道)，参数量更大，更容易拟合数据。
    """
    def __init__(self, num_classes=7, pretrained=False, **kwargs):
        super(SimpleCNN, self).__init__()
        
        # Block 1: 基础特征 (64通道) - 感受野扩大
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 尺寸减半
        )

        # Block 2: 进阶特征 (128通道)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 3: 高级特征 (256通道) - 双层卷积，提取能力加倍
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 4: 语义特征 (512通道)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 分类头
        # 无论输入图片多大，这里都会池化成 1x1，不用担心尺寸报错
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # 两层全连接，增加非线性分类能力
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # 稍微加点 Dropout，防止以后过拟合
        self.fc2 = nn.Linear(256, num_classes)

        # 初始化权重 (解决欠拟合的关键之一)
        self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)