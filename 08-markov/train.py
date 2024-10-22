import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 获取项目的根目录
import sys
import os
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)

from markovmlp import MarkovMLP
from datasets.sequential.gennerate_sin_samples import get_sin_dataloaders
from env.cuda import get_device


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()          # 清空梯度
        outputs = model(features)      # 前向传播
        outputs = outputs.squeeze()    # 调整输出形状为 [batch_size]
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数

    total_loss += loss.item() * features.size(0)  # 累加损失
    avg_loss = total_loss / len(train_loader.dataset)  # 计算平均损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')


def test(model, device, test_loader, criterion):
    # 在测试集上评估模型
    model.eval()  # 设置模型为评估模式
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            total_test_loss += loss.item() * features.size(0)
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}')


if __name__ == "__main__":
    # 超参数设置
    seq_length = 30        # tau，序列长度
    batch_size = 32       # 每个批次的样本数
    num_epochs = 200       # 训练的轮数
    learning_rate = 0.001 # 学习率

    # 获取训练和测试数据加载器
    num_samples = 300    # 数据样本总数
    sample_range = (0, 4*np.pi)  # 数据范围
    train_loader, test_loader = get_sin_dataloaders(seq_length, num_samples, batch_size, rate=0.8, sample_range=sample_range)

    # 初始化模型
    model = MarkovMLP(seq_length)

    # 检查是否可以使用 GPU
    device = get_device()
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
    
    # 预测后续正弦序列
    import matplotlib.pyplot as plt
    
    num_predictions = 200  # 定义要预测的总点数

    # 生成新的正弦波（范围可以超过训练数据范围）
    datarange=(sample_range[0]+np.pi, sample_range[1]+np.pi)  # 整体偏移一个π
    x = np.linspace(datarange[0], datarange[1], num_samples)
    x = x[:num_predictions+seq_length]
    y = np.sin(x)

    # 使用前 'seq_length' 个点作为初始输入序列
    initial_sequence = y[:seq_length]

    # 将 initial_sequence 转换为张量并移动到设备上
    input_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)  # 形状：[1, seq_length]

    # 存储预测值的列表
    predicted_values = []

    model.eval()  # 设置模型为评估模式

    with torch.no_grad():
        for _ in range(num_predictions):
            # 预测下一个值
            output = model(input_sequence)
            next_value = output.item()

            # 添加预测值到列表
            predicted_values.append(next_value)

            # 更新输入序列：移除第一个值，添加新的预测值
            input_sequence = input_sequence.squeeze().cpu().numpy()
            input_sequence = np.append(input_sequence[1:], next_value)
            input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # 准备绘图数据
    x_actual = x[seq_length:]  # 跳过用于初始输入的部分
    y_actual = y[seq_length:]
    x_predicted = x_actual
    predicted_values = np.array(predicted_values)

    # 绘制结果
    plt.figure(figsize=(12, 6))

    # 绘制实际的正弦波
    plt.plot(x_actual, y_actual, label='actual', color='blue')

    # 绘制模型预测值
    plt.plot(x_predicted, predicted_values, label='predict', color='red', linestyle='--')

    # 可选：绘制初始输入序列
    plt.plot(x[:seq_length], y[:seq_length], label='init_input', color='green')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title(' Using MLP to predict sin(x)')
    plt.show()





