# LeNet框架(需要改进：Sigmoid->ReLU, AvgPool->MaxPool)
    # Input---b,1,28,28
    # Conv2D---k5,p2---b,6,28,28
    # Sigmoid/ReLU
    # AvgPool---k2,s2---b,6,14,14
    # Conv2D---k5---b,16,10,10
    # Sigmoid/ReLU
    # AvgPool---k2,s2---b,16,5,5
    # Flatten---b,400
    # Linear---b,120
    # Sigmoid/ReLU
    # Linear---b,84
    # Sigmoid/ReLU
    # Linear---b,10


import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # 顺序定义层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # F.sigmoid -> nn.ReLU
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x, verbose=False):
        # # v1
        # if not verbose:
        #     x = self.relu(self.conv1(x))
        #     x = self.avgpool(x)
        #     x = self.relu(self.conv2(x))
        #     x = self.avgpool(x)
        #     x = self.flatten(x)
        #     x = self.relu(self.fc1(x))
        #     x = self.relu(self.fc2(x))
        #     x = self.fc3(x)
        #     return x
        # else:
        #     print(x.shape)
        #     x = self.relu(self.conv1(x))
        #     print(x.shape)
        #     x = self.avgpool(x)
        #     print(x.shape)
        #     x = self.relu(self.conv2(x))
        #     print(x.shape)
        #     x = self.avgpool(x)
        #     print(x.shape)
        #     x = self.flatten(x)
        #     print(x.shape)
        #     x = self.relu(self.fc1(x))
        #     print(x.shape)
        #     x = self.relu(self.fc2(x))
        #     print(x.shape)
        #     x = self.fc3(x)
        #     print(x.shape)
        #     return x
        # v2
        if not verbose:
            x = self.relu(self.conv1(x))
            x = self.maxpool(x)
            x = self.relu(self.conv2(x))
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)  # 最后一层不加激活函数
            return x
        else:
            print(f"Input: {x.shape}")
            x = self.relu(self.conv1(x))
            print(x.shape)
            x = self.maxpool(x)
            print(x.shape)
            x = self.relu(self.conv2(x))
            print(x.shape)
            x = self.maxpool(x)
            print(x.shape)
            x = self.flatten(x)
            print(x.shape)
            x = self.relu(self.fc1(x))
            print(x.shape)
            x = self.relu(self.fc2(x))
            print(x.shape)
            x = self.fc3(x)
            print(f"Output: {x.shape}")
            return x

if __name__ == "__main__":
    net = LeNet()
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    net.forward(X, True)
