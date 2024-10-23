import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import pickle
import random

# 获取项目的根目录
import sys
import os
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)

from lstm import LSTMModel
from datasets.sequential.text_preprocess import TextDataset
from env.cuda import get_device


# 训练函数（一个epoch)
def train(model, device, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    hidden = None
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if use_random_iter:
            hidden = None  # 重置隐藏状态
        else:
            if hidden is not None:
                hidden = hidden.detach()

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())  # 分离隐藏状态的梯度

        # 将 outputs 展平成二维张量
        outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size * seq_length, vocab_size]
        targets = targets.reshape(-1)  # [batch_size * seq_length]

        loss = criterion(outputs, targets)
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        
    # 每一个epoch打印一下训练损失
    avg_loss = total_loss / len(dataloader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')



def test(model, start_tokens, idx_to_token, token_to_idx, max_length=50):
    model.eval()
    input_seq = torch.tensor([token_to_idx.get(token, token_to_idx['<unk>']) for token in start_tokens], dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = start_tokens.copy()
    hidden = None

    with torch.no_grad():
        for _ in range(max_length):
            outputs, hidden = model(input_seq, hidden)
            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())
            # outputs 的形状: [batch_size, seq_length, vocab_size]
            last_output = outputs[:, -1, :]  # [batch_size, vocab_size]
            # 取概率最大的词元
            _, predicted_idx = torch.max(last_output, dim=1)  # predicted_idx 的形状: [batch_size]
            predicted_token = idx_to_token[predicted_idx.item()]
            generated_tokens.append(predicted_token)
            # 更新输入序列
            input_seq = torch.tensor([[predicted_idx.item()]], dtype=torch.long).to(device)
    return ' '.join(generated_tokens)

if __name__ == "__main__":
    # 数据集
    file_path = '/home/yutian/projects/d2l/d2l/datasets/sequential/The Time Machine.txt'
    seq_length = 35
    batch_size = 32
    min_freq = 3
    use_random_iter = True

    text_dataset = TextDataset(
        file_path=file_path, 
        min_freq=min_freq, 
        seq_length=seq_length, 
        batch_size=batch_size, 
        use_random_iter=use_random_iter)
    
    dataloader = text_dataset.dataloader
    vocab_size = text_dataset.vocab_size
    token_to_idx = text_dataset.token_to_idx
    idx_to_token = text_dataset.idx_to_token

    # 超参数
    embedding_dim = 128
    hidden_dim = 512  # 隐藏层神经元数量设置为 512
    num_layers = 1
    num_epochs = 100
    learning_rate = 0.001

    device = get_device()
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train(model, device, dataloader, optimizer, loss_fn, epoch)
    
    # 预测
    start_tokens = ['i', 'do', 'not', 'mean']
    generated_text = test(model, start_tokens, idx_to_token, token_to_idx, max_length=50)
    print(generated_text)
