import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        pass
    def forward(self, x, verbose=False):
        pass

if __name__ == "__main__":
    net = AlexNet()
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    net.forward(X, True)
