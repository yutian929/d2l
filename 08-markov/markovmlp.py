# 马尔可夫模型
# 马尔可夫模型是一种统计模型，假设未来的状态只取决于当前的状态，和过去的tau时段内的状态，即满足“马尔可夫性”。
# p(x) = p(x1) * p(x2|x1) * p(x3|x1, x2) * ... * p(xn|x1, x2, x3, ..., xn-1)


import torch
import torch.nn as nn


class MarkovMLP(nn.Module):
    def __init__(self, tau):
        super(MarkovMLP, self).__init__()
        self.l1 = nn.Linear(tau, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x