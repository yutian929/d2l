import torch
import torch.nn as nn
import torch.optim as optim

# 获取项目的根目录
import sys
import os
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)

from googlenet import GoogLeNet
from datasets.classify.download_fashion_minist import get_fashion_mnist_loaders
from env.cuda import get_device

# 训练函数(1个epoch)
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  # 先将之前积累的梯度清零
        # 前向传播
        outputs = model(data)  # 前向传播
        loss = criterion(outputs, target)  # 计算损失
        
        # 反向传播和优化
        loss.backward()  # 反向计算梯度
        optimizer.step()  # 更新参数
        
        # 统计
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}')

# 主函数
def main():
    # 设置设备
    device = get_device()
    
    # 获取数据加载器
    data_dir = '/home/yutian/projects/d2l/d2l/datasets/classify'
    train_loader, test_loader = get_fashion_mnist_loaders(data_dir, 
                                                          batch_size=32, 
                                                          num_workers=2, 
                                                          resize_size=(224, 224))
    
    # 创建模型
    model = GoogLeNet().to(device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 而Adam内部集成了动量（通过一阶矩估计），这有助于加速梯度下降,通常在训练初期表现出更快的收敛速度和更高的稳定性(见底部)。
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 20  # 你可以根据需要调整
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
