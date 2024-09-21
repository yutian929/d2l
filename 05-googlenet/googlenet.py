# GoogLeNet 的架构概述
#     输入层：224×224×3 的 RGB 图像。
#     卷积层和池化层：包括多个卷积和池化操作，用于初步特征提取。
#     Inception 模块：多个串联的 Inception 模块，负责更复杂和多尺度的特征提取。
#     辅助分类器：在部分 Inception 模块之后，添加辅助分类器以提供额外的梯度信号。
#     全局平均池化：将高维特征图压缩为全局特征向量。
#     输出层：Softmax 层，用于多类别分类。


import torch
import torch.nn as nn

# 定义 Inception 模块
class Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super(Inception, self).__init__()
        
        # 1x1 卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 卷积 + 3x3 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 卷积 + 5x5 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 最大池化 + 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # debug
        # print(branch1.shape, branch2.shape, branch3.shape, branch4.shape)
        # 拼接所有分支的输出
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# 定义 GoogLeNet（Inception v1）
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        
        # 初始卷积层和池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception 模块堆叠
        self.inception3a = Inception(192,  64,  96, 128, 16,  32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32,  96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception(480, 192,  96, 208, 16,  48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24,  64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24,  64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32,  64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [B, C, H, W] -> [B, C, 1, 1]
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(832, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# 示例：创建 GoogLeNet 实例
if __name__ == "__main__":
    model = GoogLeNet(num_classes=10)  # 例如，适用于 Fashion-MNIST
    print(model)
    # 生成一个示例输入
    input_tensor = torch.randn(1, 1, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # 应输出 torch.Size([1, 10])

