# 多GPU训练在PyTorch中非常常见。以下是一个使用torch.nn.DataParallel进行多GPU训练的简单例子。
# 代码中使用torch.nn.DataParallel将模型分配到多块GPU上。DataParallel会自动将输入数据分割并分配给各个GPU，每个GPU计算出它的梯度后再汇总到主GPU进行梯度更新。
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义逻辑回归模型（线性模型）
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear_1 = nn.Linear(1, 4)  # 输入1维
        self.linear_2 = nn.Linear(4, 1)  # 输出1维
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

# 生成线性数据集，假设斜率为2.0，偏移量为1.0，加一些噪声
def create_linear_dataset(slope=2.0, intercept=1.0, num_samples=1000):
    X = torch.randn(num_samples, 1)
    y = slope * X + intercept + 0.1 * torch.randn(num_samples, 1)  # 加入少量噪声
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
def train_model(model, train_loader, optimizer=None, criterion=None, device='cpu'):
    model.train()
    for epoch in range(10):  # 假设训练10个epoch
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

if __name__ == '__main__':
    # 初始化模型、数据集和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearRegressionModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_loader = create_linear_dataset()

    # 检查GPU数量
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU 进行训练！")
        # 使用 DataParallel 进行多GPU并行
        model = nn.DataParallel(model)

    # 开始训练
    train_model(model, train_loader, optimizer, criterion, device)

    # 输出学习到的模型参数（斜率和偏移）
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")

# 使用 8 个 GPU 进行训练！
# Epoch 1, Loss: 3.8696843422949314
# Epoch 2, Loss: 1.4894630517810583
# Epoch 3, Loss: 0.4291863045655191
# Epoch 4, Loss: 0.15322698024101555
# Epoch 5, Loss: 0.11782467481680214
# Epoch 6, Loss: 0.09989156678784639
# Epoch 7, Loss: 0.09032668988220394
# Epoch 8, Loss: 0.08547797019127756
# Epoch 9, Loss: 0.08398319751722738
# Epoch 10, Loss: 0.08187113213352859
# module.linear_1.weight: tensor([[ 0.3604],
#         [-0.3350],
#         [-1.0605],
#         [ 1.2158]], device='cuda:0')
# module.linear_1.bias: tensor([-0.5348,  0.4725,  0.2373,  0.8937], device='cuda:0')
# module.linear_2.weight: tensor([[-0.0853, -0.3680, -1.0085,  1.3093]], device='cuda:0')
# module.linear_2.bias: tensor([0.3645], device='cuda:0')