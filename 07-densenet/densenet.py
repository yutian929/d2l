# 1. DenseNet 的核心思想
# 1.1. 密集连接（Dense Connections）
# DenseNet 的核心创新在于每一层都与前面所有层直接连接。这意味着，对于网络中的每一层，其输入不仅包括前一层的输出，还包括所有之前层的输出。这种设计带来了以下几个优势：
#     特征复用（Feature Reuse）：每一层都可以访问之前所有层的特征，这促进了特征的复用，减少了冗余。
#     梯度流动增强：密集连接有助于缓解梯度消失问题，使得更深的网络更容易训练。
#     参数效率高：由于特征复用，DenseNet 可以在较少的参数下实现高性能。
# 1.2. 密集块（Dense Blocks）和过渡层（Transition Layers）
# DenseNet 的架构由多个密集块和过渡层交替组成：
#     密集块（Dense Blocks）：由多个密集连接的卷积层组成。在一个密集块内，每一层接收前面所有层的特征图作为输入，并将自己的特征图传递给后续所有层。
#     过渡层（Transition Layers）：位于密集块之间，用于控制特征图的尺寸和数量。过渡层通常包含 1×1 卷积和 2×2 平均池化，用于减少特征图的数量和尺寸。
# 1.3. 特征增长率（Growth Rate）
# 特征增长率是 DenseNet 的一个重要超参数，表示每个密集块中每层输出特征图的数量。较小的增长率可以显著减少参数数量，同时保持模型性能。



import torch
import torch.nn as nn


# 定义密集块（Dense Block）
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        """
        num_layers：每个密集块中的卷积层数量，3层的话最终就会在每个像素上concat3次通道值
        in_channels：最初始的输入到密集块的特征图的通道数。
        growth_rate：每一层生成的新特征图的通道数。它决定了网络的“增长”速度
        
        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)
    
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        features = [x]  # 这个列表将存储所有之前的特征图，以便后续层可以连接这些特征。
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))  # 将当前特征列表中的所有特征图在通道维度上进行拼接。dim=1 表示沿着通道维度拼接。
            features.append(new_feat)
        # 最终，密集块的输出是所有层的特征图在通道维度上的拼接，通道数为 in_channels + num_layers * growth_rate。
        return torch.cat(features, dim=1)


# 定义过渡层（Transition Layer）
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 主要就是通过1x1卷积，将通道数减少来降低复杂度
            nn.AvgPool2d(kernel_size=2, stride=2)  # 高宽也减半
        )
    
    def forward(self, x):
        return self.transition(x)


# 定义简化版 DenseNet
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_layers=[4, 4, 4], num_classes=10):
        super(DenseNet, self).__init__()
        num_init_features = 64
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),  # Fashion-MNIST 是灰度图
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
            # 由于输入尺寸较小，不使用池化层
        )
        
        # 添加密集块和过渡层
        in_channels = num_init_features
        for i, num_layers in enumerate(block_layers):
            # 添加密集块
            block = DenseBlock(num_layers=num_layers, in_channels=in_channels, growth_rate=growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            in_channels += num_layers * growth_rate
            
            # 添加过渡层（除最后一个密集块外）
            if i != len(block_layers) - 1:
                out_channels = in_channels // 2  # 压缩特征图数量
                transition = TransitionLayer(in_channels, out_channels)
                self.features.add_module(f'transition{i+1}', transition)
                in_channels = out_channels
        
        # 最后的 BatchNorm
        self.features.add_module('norm_final', nn.BatchNorm2d(in_channels))
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.classifier = nn.Linear(in_channels, num_classes)
        
        # # 初始化权重
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    db = DenseBlock(num_layers=2, in_channels=3, growth_rate=10)
    print(db(x).shape)

    x = torch.randn(1, 23, 32, 32)
    tl = TransitionLayer(23, 16)
    print(tl(x).shape)

    x = torch.randn(1, 1, 32, 32)
    densenet = DenseNet()
    print(densenet(x).shape)