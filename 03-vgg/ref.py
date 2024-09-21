import torch
import torch.nn as nn

class VGGMinimal(nn.Module):
    def __init__(self, num_classes=10):
        """
        简化版 VGG 模型（VGG9）
        
        参数：
        - num_classes: 分类的类别数，默认为 10（适用于 Fashion-MNIST）。
        """
        super(VGGMinimal, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 输入通道数：1，输出通道数：32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 图像尺寸减半：64x64 -> 32x32
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 输入通道数：32，输出通道数：64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 图像尺寸减半：32x32 -> 16x16
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 输入通道数：64，输出通道数：128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 图像尺寸减半：16x16 -> 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # 全连接层：128*8*8 = 8192 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),             # Dropout 层，防止过拟合
            nn.Linear(256, num_classes)    # 输出层：256 -> num_classes
        )
        
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
        
        x = torch.flatten(x, 1)  # 展平为 [B, 128*8*8 = 8192]
        if verbose:
            print(f"After flatten: {x.shape}")  # [B, 8192]
        
        x = self.classifier(x)
        if verbose:
            print(f"After classifier: {x.shape}")  # [B, num_classes]
        
        return x

# 测试模型
if __name__ == "__main__":
    model = VGGMinimal(num_classes=10)
    X = torch.rand(size=(1, 1, 64, 64), dtype=torch.float32)  # 调整为 64x64
    output = model.forward(X, verbose=True)
    print(f"Output: {output}")
