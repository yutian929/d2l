import torch
import torch.nn as nn
import torch.optim as optim

# 获取项目的根目录
import sys
import os
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)

from nin import NiN
# from ref import NiN
from datasets.classify.download_fashion_minist import get_fashion_mnist_loaders
from env.cuda import get_device

# # 训练函数(1个epoch)
# def train(model, device, train_loader, optimizer, criterion, epoch):
#     model.train()
#     running_loss = 0.0
#     total = 0
#     correct = 0
    
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad()  # 先将之前积累的梯度清零
#         # 前向传播
#         outputs = model(data)  # 前向传播
#         loss = criterion(outputs, target)  # 计算损失
        
#         # 反向传播和优化
#         loss.backward()  # 反向计算梯度
#         optimizer.step()  # 更新参数
        
#         # 统计
#         running_loss += loss.item() * data.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         total += target.size(0)
#         correct += (predicted == target).sum().item()
        
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     print(f'Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

# # 测试函数
# def test(model, device, test_loader, criterion):
#     model.eval()
#     running_loss = 0.0
#     total = 0
#     correct = 0
    
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
            
#             outputs = model(data)
#             loss = criterion(outputs, target)
            
#             running_loss += loss.item() * data.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
            
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     print(f'Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}')

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
    
    # 获取数据加载器
    data_dir = '/home/yutian/projects/d2l/d2l/datasets/classify'
    train_loader, test_loader = get_fashion_mnist_loaders(data_dir, 
                                                          batch_size=64, 
                                                          num_workers=2, 
                                                          resize_size=(64, 64))
    
    # 创建模型
    model = NiN().to(device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 为什么这里不用SGD？SGD对学习率敏感，而且前几个epoch都没啥反映(见底部)，所以不用。
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 而Adam内部集成了动量（通过一阶矩估计），这有助于加速梯度下降,通常在训练初期表现出更快的收敛速度和更高的稳定性(见底部)。
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 20  # 你可以根据需要调整
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion, epoch)

if __name__ == '__main__':
    main()

# 使用SGD的打印输出:
# Epoch 1: Train Loss: 2.3032, Train Acc: 0.0993
# Epoch 1: Test Loss: 2.3029, Test Acc: 0.1000

# Epoch 2: Train Loss: 2.3032, Train Acc: 0.0989
# Epoch 2: Test Loss: 2.3029, Test Acc: 0.1000

# Epoch 3: Train Loss: 2.3032, Train Acc: 0.0987
# Epoch 3: Test Loss: 2.3029, Test Acc: 0.1000

# Epoch 4: Train Loss: 2.3031, Train Acc: 0.0989
# Epoch 4: Test Loss: 2.3029, Test Acc: 0.1000

# Epoch 5: Train Loss: 2.3031, Train Acc: 0.0985
# Epoch 5: Test Loss: 2.3029, Test Acc: 0.1000

# Epoch 6: Train Loss: 2.3031, Train Acc: 0.0988
# Epoch 6: Test Loss: 2.3029, Test Acc: 0.1000

# Epoch 7: Train Loss: 2.3031, Train Acc: 0.0988
# Epoch 7: Test Loss: 2.3028, Test Acc: 0.1000

# Epoch 8: Train Loss: 2.3028, Train Acc: 0.0999
# Epoch 8: Test Loss: 2.3014, Test Acc: 0.1000

# Epoch 9: Train Loss: 1.3381, Train Acc: 0.4924
# Epoch 9: Test Loss: 0.6719, Test Acc: 0.7488

# Epoch 10: Train Loss: 0.6236, Train Acc: 0.7719
# Epoch 10: Test Loss: 0.5653, Test Acc: 0.7980

# Epoch 11: Train Loss: 0.5175, Train Acc: 0.8133
# Epoch 11: Test Loss: 0.4853, Test Acc: 0.8249

# Epoch 12: Train Loss: 0.4558, Train Acc: 0.8352
# Epoch 12: Test Loss: 0.4276, Test Acc: 0.8478

# Epoch 13: Train Loss: 0.4102, Train Acc: 0.8498
# Epoch 13: Test Loss: 0.4054, Test Acc: 0.8578

# Epoch 14: Train Loss: 0.3798, Train Acc: 0.8632
# Epoch 14: Test Loss: 0.3641, Test Acc: 0.8734

# 使用Adam的打印输出:
# Epoch 1: Train Loss: 0.8185, Train Acc: 0.6884
# Epoch 1: Test Loss: 0.5366, Test Acc: 0.7966

# Epoch 2: Train Loss: 0.4873, Train Acc: 0.8183
# Epoch 2: Test Loss: 0.4171, Test Acc: 0.8433

# Epoch 3: Train Loss: 0.4002, Train Acc: 0.8532
# Epoch 3: Test Loss: 0.3721, Test Acc: 0.8588

# Epoch 4: Train Loss: 0.3491, Train Acc: 0.8722
# Epoch 4: Test Loss: 0.3394, Test Acc: 0.8746

# Epoch 5: Train Loss: 0.3151, Train Acc: 0.8840
# Epoch 5: Test Loss: 0.3304, Test Acc: 0.8786

# Epoch 6: Train Loss: 0.2903, Train Acc: 0.8931
# Epoch 6: Test Loss: 0.3075, Test Acc: 0.8855

# Epoch 7: Train Loss: 0.2717, Train Acc: 0.8997
# Epoch 7: Test Loss: 0.2954, Test Acc: 0.8904

# Epoch 8: Train Loss: 0.2609, Train Acc: 0.9040
# Epoch 8: Test Loss: 0.3062, Test Acc: 0.8857

# Epoch 9: Train Loss: 0.2470, Train Acc: 0.9093
# Epoch 9: Test Loss: 0.2806, Test Acc: 0.8986

# Epoch 10: Train Loss: 0.2370, Train Acc: 0.9122
# Epoch 10: Test Loss: 0.2549, Test Acc: 0.9081

# Epoch 11: Train Loss: 0.2258, Train Acc: 0.9174
# Epoch 11: Test Loss: 0.2769, Test Acc: 0.9001