import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 定义简化版的NiN网络
class NiN(nn.Module):
    def __init__(self, num_classes=10):
        """
        初始化 NiN 网络。
        
        参数：
        - num_classes: 分类的类别数，默认为 10（适用于 Fashion-MNIST）。
        """
        super(NiN, self).__init__()
        
        # 特征提取部分，由多个 NiN 块组成
        self.features = self._make_layers()
        
        # 分类器部分，使用 1x1 卷积代替全连接层
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=0),  # 卷积层，kernel_size=5，无填充
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 128, kernel_size=1, padding=0),  # 1x1 卷积层
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_classes, kernel_size=1, padding=0),  # 输出类别数的 1x1 卷积层
            # nn.ReLU(inplace=True),  # 移除最后的 ReLU
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，将特征图尺寸缩减为 1x1
        )
    
    def NiNBlock(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        定义一个 NiN 块，包括一个标准卷积层后跟两个 1x1 卷积层。
        
        参数：
        - in_channels: 输入通道数。
        - out_channels: 输出通道数。
        - kernel_size: 标准卷积层的卷积核大小，默认为 3。
        - stride: 标准卷积层的步幅，默认为 1。
        - padding: 标准卷积层的填充，默认为 1。
        
        返回：
        - nn.Sequential: 由卷积层、激活函数和池化层组成的 NiN 块。
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样，减半特征图尺寸
        )
    
    def _make_layers(self):
        """
        构建特征提取部分，通过多个 NiN 块堆叠。
        
        返回：
        - nn.Sequential: 由多个 NiN 块组成的特征提取部分。
        """
        layers = []
        # 第一层 NiN 块：输入通道数 1，输出通道数 32
        layers.append(self.NiNBlock(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2))
        # 第二层 NiN 块：输入通道数 32，输出通道数 64
        layers.append(self.NiNBlock(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2))
        # 第三层 NiN 块：输入通道数 64，输出通道数 128
        layers.append(self.NiNBlock(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2))
        
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
            print(f"Input shape: {x.shape}")  # 例如：[B, 1, 64, 64]
        
        x = self.features(x)
        if verbose:
            print(f"After features: {x.shape}")  # 例如：[B, 128, 8, 8]
        
        x = self.classifier(x)
        if verbose:
            print(f"After classifier: {x.shape}")  # 例如：[B, num_classes, 1, 1]
        
        x = x.view(x.size(0), -1)  # 展平为 [B, num_classes]
        if verbose:
            print(f"After flatten: {x.shape}")  # 例如：[B, num_classes]
        
        return x


# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    训练模型一个 epoch，并打印训练损失和准确率。
    
    参数：
    - model: 要训练的模型。
    - device: 设备（CPU 或 GPU）。
    - train_loader: 训练数据的 DataLoader。
    - optimizer: 优化器。
    - criterion: 损失函数。
    - epoch: 当前的 epoch 数。
    
    返回：
    - None
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 将梯度归零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计损失
        running_loss += loss.item() * data.size(0)
        
        # 统计准确率
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')


# 测试函数
def test(model, device, test_loader, criterion, epoch):
    """
    测试模型，并打印测试损失和准确率。
    
    参数：
    - model: 要测试的模型。
    - device: 设备（CPU 或 GPU）。
    - test_loader: 测试数据的 DataLoader。
    - criterion: 损失函数。
    
    返回：
    - None
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 测试阶段不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 统计损失
            running_loss += loss.item() * data.size(0)
            
            # 统计准确率
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 计算平均损失和准确率
    test_loss = running_loss / total
    test_acc = correct / total
    print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n')


# 主函数
def main():
    # 设置设备
    device = get_device()

    # 加载Fashion-MNIST数据集
    data_dir = '/home/yutian/projects/d2l/d2l/datasets/classify'
    train_loader, test_loader = get_fashion_mnist_loaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=2,
        resize_size=(64, 64)
    )
    
    # 创建模型并移动到设备
    model = NiN().to(device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 20  # 你可以根据需要调整
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion, epoch)
    
    print('Finished Training')


if __name__ == "__main__":
    # 调试用
    import sys
    import os
    project_root = "/home/yutian/projects/d2l/d2l"
    sys.path.append(project_root)

    from datasets.classify.download_fashion_minist import get_fashion_mnist_loaders
    from env.cuda import get_device
    main()
