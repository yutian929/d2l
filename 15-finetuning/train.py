import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    # 图像增广和预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),          # 随机裁剪并缩放到224x224
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet的均值和标准差
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),                     # 调整大小
        transforms.CenterCrop(224),                 # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet的均值和标准差
    ])

    # 加载训练和测试数据集
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def initialize_model():
    # 加载预训练的ResNet-18模型
    model = models.resnet18(pretrained=True)
    # 修改最后一层，全连接层以适应二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类：热狗和非热狗
    return model

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 每个epoch输出损失和准确率
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def test_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据目录和批次大小
    data_dir = "/mnt/zhangyutian/projects/finished/task10_yutian/d2l/d2l/datasets/classify/HotdogData/hotdog"
    batch_size = 32

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)

    # 初始化模型
    model = initialize_model()
    model = model.to(device)  # 将模型加载到GPU（如果可用）

    # 使用 DataParallel 包装模型，实现多GPU并行
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    print("Starting Training...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)

    # 测试模型
    print("Starting Testing...")
    test_model(model, test_loader, device)

# Using 8 GPUs!
# Starting Training...
# Epoch 1/5, Loss: 0.5649, Accuracy: 70.15%
# Epoch 2/5, Loss: 0.4631, Accuracy: 78.25%
# Epoch 3/5, Loss: 0.4556, Accuracy: 78.25%
# Epoch 4/5, Loss: 0.4451, Accuracy: 78.60%
# Epoch 5/5, Loss: 0.4301, Accuracy: 79.95%
# Starting Testing...
# Test Accuracy: 92.00%
