# 1. 模型结构

# Seq2Seq模型的基本架构是编码器-解码器（encoder-decoder）结构：

#     编码器 (Encoder)：将输入的句子编码为一个上下文向量（隐状态），通常使用LSTM或GRU单元。
#     解码器 (Decoder)：接收编码器输出的上下文向量，生成目标语言的翻译结果。解码器通常也是LSTM或GRU。
#     注意力机制 (Attention)（可选）：通过注意力机制增强模型，使解码器在每一步都能关注编码器输出的不同部分。这在处理较长句子时尤其有帮助。

# 2. 步骤详解

#     定义模型类
#         编码器：接收词汇索引序列，将其嵌入成词向量，再通过LSTM/GRU生成隐状态。
#         解码器：在每一步生成词汇的概率分布，预测下一个词。
#         注意力机制（可选）：如果添加Attention机制，可以在解码器中计算每一步的上下文向量。

#     训练过程
#         逐词计算损失函数。
#         使用Teacher Forcing技术（即用目标词而不是模型预测的词作为解码器的输入）来加速模型训练。
#         使用交叉熵损失来计算解码器的输出和目标翻译句子的损失。

#     推理过程
#         推理时，解码器逐步生成目标词，每一步将上一步生成的词作为下一步的输入，直到遇到<eos>标记。
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input: [batch_size]
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]
        prediction = self.out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder hidden state
        hidden = self.encoder(src)
        
        # First input to the decoder is the <bos> tokens
        input = trg[:, 0]  # [batch_size]
        
        for t in range(1, trg_len):
            # Pass through the decoder
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            # Decide whether to do teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            # Get the highest predicted token from output
            top1 = output.argmax(1)
            # If teacher forcing, use actual next token as next input; else use predicted token
            input = trg[:, t] if teacher_force else top1
        
        return outputs
