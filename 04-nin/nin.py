#  NiN 的关键特点
#     MLP 卷积层（1x1 卷积）：通过 1x1 卷积引入非线性变换，使每个局部感受野内的特征能够进行更复杂的组合。
#     全局平均池化（Global Average Pooling, GAP）：在网络末端使用全局平均池化替代全连接层，减少参数数量，防止过拟合，并直接输出每个类别的概率。
#     更深的非线性表达：相比传统的 CNN，NiN 在每个卷积块中增加了更多的非线性变换，提升了模型的表达能力。


import torch
import torch.nn as nn

class NiN(nn.Module):
    def __init__(self, conv_arch=None, num_classes=10):
        """
        NiN 模型初始化函数。

        参数：
        - conv_arch: list of tuples [(num_convs, in_channels, out_channels), ...]
                     每个元组定义一个 NiN 块，包括卷积层数量、输入通道数和输出通道数。
                     默认值为简化版 NiN 架构。
        - num_classes: 分类的类别数，默认为 10（适用于 Fashion-MNIST）。
        """
        super(NiN, self).__init__()

        if conv_arch is None:
            # 默认简化版 NiN 架构（3 个 NiN 块）
            conv_arch = (
                (1, 32),   # 第一个 NiN 块：输入通道数 1，输出通道数 32
                (32, 64),  # 第二个 NiN 块：输入通道数 32，输出通道数 64
                (64, 128)  # 第三个 NiN 块：输入通道数 64，输出通道数 128
            )
        self.conv_arch = conv_arch

        # 创建特征提取部分
        self.features = self._make_layers(self.conv_arch)

        # 分类器部分
        # 输入特征数根据输入图像尺寸和网络结构计算
        # 输入图像尺寸 64x64，经过 3 次 2x2 最大池化，特征图尺寸为 8x8
        # 最后一个 NiN 块输出通道数为 128，因此全连接层输入特征数为 128 * 8 * 8 = 8192
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_classes, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )

    def _make_layers(self, conv_arch):
        """
        根据卷积架构创建特征提取部分。

        参数：
        - conv_arch: list of tuples [(num_convs, in_channels, out_channels), ...]

        返回：
        - nn.Sequential: 由多个 NiN 块组成的特征提取部分。
        """
        layers = []
        for in_channels, out_channels in conv_arch:
            layers.append(self.NiNBlock(in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def NiNBlock(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        NiN 块的定义，包括一个卷积层和两个 1x1 卷积层。

        参数：
        - in_channels: 输入通道数。
        - out_channels: 输出通道数。
        - kernel_size: 卷积核大小，默认为 3。
        - stride: 步幅，默认为 1。
        - padding: 填充，默认为 1。
        """
        nin_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，用于下采样
        )
        return nin_block


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

        x = self.classifier(x)
        if verbose:
            print(f"After classifier: {x.shape}")  # [B, num_classes, 1, 1]

        x = x.view(x.size(0), -1)  # 展平为 [B, num_classes]
        if verbose:
            print(f"After flatten: {x.shape}")  # [B, num_classes]

        return x

# 测试模型
if __name__ == "__main__":
    conv_arch = (
        (1, 32),
        (32, 64),
        (64, 128)
    )
    net = NiN(conv_arch, num_classes=10)
    X = torch.rand(size=(1, 1, 64, 64), dtype=torch.float32)  # 调整为 64x64
    output = net.forward(X, True)
    print(f"Output: {output}")
