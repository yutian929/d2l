import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)  # [batch_size, seq_length, vocab_size]
        return output, hidden

