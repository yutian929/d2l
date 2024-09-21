# ResNet 的关键特点
# 1. 深层架构的突破
# 在 ResNet 之前，随着网络深度的增加，模型的性能并未持续提升，甚至出现了退化问题（degradation problem），即更深的网络在训练和测试时的准确率反而下降。ResNet 通过引入 残差学习（Residual Learning） 解决了这一问题，使得构建上百层甚至千层的深度网络成为可能。
# 2. 残差块（Residual Block）
# ResNet 的核心创新在于 残差块，它通过引入跳跃连接（skip connections）将输入直接传递到后续层，形成所谓的“捷径路径”。这种设计允许网络学习残差函数，而不是直接学习目标函数。
# 残差块的结构

# 一个典型的残差块包括两个主要部分：
#     主路径（Main Path）：包含一系列卷积层、批量归一化（Batch Normalization）和激活函数（通常是 ReLU）。
#     捷径路径（Shortcut Path）：直接将输入添加到主路径的输出。
#       Input
#       |
#       [Conv -> BN -> ReLU -> Conv -> BN]
#       |                          |
#       |----------+---------------|
#                   |
#               Addition
#                   |
#               ReLU
#                   |
#               Output
#                   |


import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # downsample 用于调整输入和输出的维度
        if downsample is not None:
            if downsample == True:  # 使用默认的下采样，及使用1x1卷积，高宽减半
                assert stride == 2, "downsample is True but stride != 2"
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=2, bias=False),
                    nn.BatchNorm2d(out_channels))
            else:  # 使用自定义的下采样, 需要用户自行计算stride
                self.downsample = downsample
        else:  # 不使用下采样, 则需要保证输入和输出维度相同， 即 stride=1
            assert stride == 1, "downsample is None but stride != 1"
            self.downsample = downsample  # 用于匹配输入和输出的维度

        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入以供跳跃连接使用

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 如果需要下采样，则调整身份路径的维度
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):

        super(ResNet, self).__init__()
        
        # 初始输入要经过的卷积层（调整为适应 28x28 输入, 输出为 64 通道）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Fashion-MNIST 是灰度图，输入通道数为1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 为了小尺寸输入，不使用最大池化层
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        # 第一层是两个残差块， 不改变图像高宽和通道数
        self.layer1 = self._make_layer(64, 64)
        # 第二、三、四层都分别由两个残差块组成， 并且高宽减半，通道数加倍
        self.layer2 = self._make_layer(64, 128, downsample=True, stride=2)
        self.layer3 = self._make_layer(128, 256, downsample=True, stride=2)
        self.layer4 = self._make_layer(256, 512, downsample=True, stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # 最后一层的输出通道数为512

        # # 初始化权重
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, downsample=None, stride=1, blocks=2):
        # 基本都是两个残差块构成一个残差层
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, downsample, stride))
        layers.append(ResidualBlock(out_channels, out_channels))  # 第二层不改变任何东西
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)      # [B, 64, 28, 28]
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)   # 不使用池化层
        
        x = self.layer1(x)     # [B, 64, 28, 28]
        x = self.layer2(x)     # [B, 128, 14, 14]
        x = self.layer3(x)     # [B, 256, 7, 7]
        x = self.layer4(x)     # [B, 512, 4, 4]
        
        x = self.avgpool(x)    # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)         # [B, num_classes]
        
        return x


if __name__ == "__main__":
    # 输入等于输出，不需要下采样，stride=1，图片高宽不变
    blk = ResidualBlock(in_channels=1, out_channels=1, stride=1, downsample=None)
    X = torch.rand(1, 1, 28, 28)
    Y = blk(X)
    print(Y.shape)

    # 输入不等于输出，需要下采样，stride=2，图片高宽减半
    blk = ResidualBlock(in_channels=1, out_channels=6, downsample=True, stride=2)
    X = torch.rand(1, 1, 28, 28)
    Y = blk(X)
    print(Y.shape)

    # 测试ResNet:
    net = ResNet()
    print(net)

    # 测试模型
    X = torch.rand(1, 1, 28, 28)
    Y = net(X)
    print(Y.shape)