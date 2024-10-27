import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def encode(self, src, src_key_padding_mask):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer.encoder(src_emb.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        output = self.transformer.decoder(
            tgt_emb.transpose(0, 1),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return output

    def forward(self, src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        memory = self.encode(src, src_key_padding_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask)
        output = self.generator(output.transpose(0, 1))
        return output
