# VGG 的关键特点
#     统一的卷积核大小：VGG 使用了多个连续的 3×33×3 卷积核，这些小卷积核能够增加网络的深度，同时保持感受野的大小。
#     深度：VGG 模型通常较深，如 VGG16 有 16 层（13 层卷积层和 3 层全连接层），VGG19 有 19 层。
#     最大池化：在每组卷积层后使用 2×22×2 的最大池化层，逐步减少特征图的尺寸。
#     全连接层：在卷积层之后，使用三个全连接层，其中前两个有 4096 个神经元，最后一层的输出节点数对应于分类任务的类别数。


import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, conv_arch=None, num_classes=10):
        """
        VGG 模型初始化函数。

        参数：
        - conv_arch: list of tuples [(num_convs, in_channels, out_channels), ...]
                     每个元组定义一个 VGG 块，包括卷积层数量、输入通道数和输出通道数。
                     默认值为简化版 VGG 架构。
        - num_classes: 分类的类别数，默认为 10(适用于 Fashion-MNIST)。
        """
        super(VGG, self).__init__()

        if conv_arch is None:
            # 默认简化版 VGG 架构（3 个卷积块）
            conv_arch = (
                (1, 1, 32),   # 第一个卷积块：1 个卷积层，输入通道数 1，输出通道数 32
                (1, 32, 64),  # 第二个卷积块：1 个卷积层，输入通道数 32，输出通道数 64
                (2, 64, 128)  # 第三个卷积块：2 个卷积层，输入通道数 64，输出通道数 128
            )
        self.conv_arch = conv_arch

        # 创建特征提取部分
        self.features = self._make_layers(self.conv_arch)

        # 分类器部分
        # 输入特征数根据输入图像尺寸和网络结构计算
        # 输入图像尺寸 64x64，经过 3 次 2x2 最大池化，特征图尺寸为 8x8
        # 最后一个卷积块输出通道数为 128，因此全连接层输入特征数为 128 * 8 * 8 = 8192
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # 全连接层：8192 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),            # Dropout 层，防止过拟合
            nn.Linear(256, 128),          # 全连接层：256 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),            # Dropout 层，防止过拟合
            nn.Linear(128, num_classes)   # 输出层：128 -> num_classes
        )

    def _make_layers(self, conv_arch):
        """
        根据卷积架构创建特征提取部分。

        参数：
        - conv_arch: list of tuples [(num_convs, in_channels, out_channels), ...]

        返回：
        - nn.Sequential: 由多个 VGG 块组成的特征提取部分。
        """
        layers = []
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            layers.append(self.vgg_block(num_convs, in_channels, out_channels))
        return nn.Sequential(*layers)

    def vgg_block(self, num_convs, in_channels, out_channels):
        """
        创建一个 VGG 块，包括多个卷积层和一个最大池化层。

        参数：
        - num_convs: 卷积层的数量。
        - in_channels: 输入通道数。
        - out_channels: 输出通道数。

        返回：
        - nn.Sequential: 包含多个卷积层、ReLU 激活和最大池化层的模块。
        """
        layers = []
        for _ in range(num_convs):
            # 添加卷积层，使用 3x3 的卷积核，保持特征图的高宽不变
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            # 添加 ReLU 激活函数
            layers.append(nn.ReLU(inplace=True))
            # 更新输入通道数为当前卷积层的输出通道数
            in_channels = out_channels
        # 添加最大池化层，使用 2x2 的池化核，步长为 2，使特征图的高宽减半
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x, verbose=False):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 [B, C, H, W]。
        - verbose: 是否打印中间层的输出形状，默认为 False。

        返回：
        - 输出张量，形状为 [B, num_classes]。
        """
        if verbose:
            print(f"Input shape: {x.shape}")  # [B, 1, 64, 64]

        x = self.features(x)
        if verbose:
            print(f"After features: {x.shape}")  # [B, 128, 8, 8]

        x = torch.flatten(x, 1)  # 展平，保留 batch 维度
        if verbose:
            print(f"After flatten: {x.shape}")  # [B, 8192]

        x = self.classifier(x)
        if verbose:
            print(f"After classifier: {x.shape}")  # [B, num_classes]

        return x

# 测试模型
if __name__ == "__main__":
    conv_arch = (
        (1, 1, 32),   # 第一个卷积块：1 个卷积层，输入通道数 1，输出通道数 32
        (1, 32, 64),  # 第二个卷积块：1 个卷积层，输入通道数 32，输出通道数 64
        (2, 64, 128)  # 第三个卷积块：2 个卷积层，输入通道数 64，输出通道数 128
    )
    net = VGG(conv_arch, num_classes=10)
    X = torch.rand(size=(1, 1, 64, 64), dtype=torch.float32)  # 调整为 64x64
    output = net.forward(X, True)
    print(f"Output: {output}")
