import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义 LeNet 网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # 定义网络层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层不加激活函数
        return x

# 获取 Fashion-MNIST 数据加载器
def get_fashion_mnist_loaders(data_dir, batch_size=64, num_workers=2):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 下载并加载训练集和测试集
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    
    # 获取数据加载器
    data_dir = '/home/yutian/projects/d2l/d2l/datasets/classify'
    train_loader, test_loader = get_fashion_mnist_loaders(data_dir, batch_size=64, num_workers=2)
    
    # 创建模型
    model = LeNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 5  # 可以根据需要调整
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)
        
if __name__ == '__main__':
    main()
