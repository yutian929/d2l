# LSTM 的基本原理

#     记忆单元（Cell State）：LSTM 的核心是细胞状态，它像传送带一样，在序列中传递信息。信息可以在细胞状态中流动，受到最小程度的修改，从而保持长期的依赖信息。

#     门控机制：
#         遗忘门（Forget Gate）：决定应该从细胞状态中遗忘哪些信息。
#         输入门（Input Gate）：决定哪些新的信息需要添加到细胞状态中。
#         输出门（Output Gate）：决定从细胞状态中输出哪些信息。

# 通过这些门控，LSTM 可以选择性地记忆或遗忘信息，从而在处理长序列时保持稳定的梯度。


import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 使用 LSTM 而不是 RNN
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)  # [batch_size, seq_length, vocab_size]
        return output, hidden
