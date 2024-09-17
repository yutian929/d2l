import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
        # AlexNet 框架（适用于灰度图像）
        # Input: b,1,224,224
        # Conv1: k11,p2,s4 -> b,96,55,55
        # ReLU
        # MaxPool1: k3,s2 -> b,96,27,27
        # Conv2: k5,p2,s1 -> b,256,27,27
        # ReLU
        # MaxPool2: k3,s2 -> b,256,13,13
        # Conv3: k3,p1,s1 -> b,384,13,13
        # ReLU
        # Conv4: k3,p1,s1 -> b,384,13,13
        # ReLU
        # Conv5: k3,p1,s1 -> b,256,13,13
        # ReLU
        # MaxPool3: k3,s2 -> b,256,6,6
        # Flatten: b,256*6*6
        # Linear1: b,4096
        # ReLU
        # Linear2: b,4096
        # ReLU
        # Linear3: b,10
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)  # 假设分类数为10
        
    def forward(self, x, verbose=False):
        if verbose:
            print(f"Input shape: {x.shape}")
        
        # 第一层卷积
        x = self.relu(self.conv1(x))
        if verbose:
            print(f"After conv1 and relu: {x.shape}")
        x = self.maxpool(x)
        if verbose:
            print(f"After maxpool1: {x.shape}")
        
        # 第二层卷积
        x = self.relu(self.conv2(x))
        if verbose:
            print(f"After conv2 and relu: {x.shape}")
        x = self.maxpool(x)
        if verbose:
            print(f"After maxpool2: {x.shape}")
        
        # 第三层卷积
        x = self.relu(self.conv3(x))
        if verbose:
            print(f"After conv3 and relu: {x.shape}")
        
        # 第四层卷积
        x = self.relu(self.conv4(x))
        if verbose:
            print(f"After conv4 and relu: {x.shape}")
        
        # 第五层卷积
        x = self.relu(self.conv5(x))
        if verbose:
            print(f"After conv5 and relu: {x.shape}")
        x = self.maxpool(x)
        if verbose:
            print(f"After maxpool3: {x.shape}")
        
        # 展平
        x = self.flatten(x)
        if verbose:
            print(f"After flatten: {x.shape}")
        
        # 全连接层
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        if verbose:
            print(f"After fc1 and relu: {x.shape}")
        
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if verbose:
            print(f"After fc2 and relu: {x.shape}")
        
        x = self.fc3(x)
        if verbose:
            print(f"Output shape: {x.shape}")
        
        return x

if __name__ == "__main__":
    net = AlexNet()
    X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
    net.forward(X, True)
