import math
import torch
import torch.nn as nn
import torch.optim as optim


# Positional Encoding
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

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute scaled dot-product attention
    q, k, v: tensors of shape (batch_size, num_heads, seq_len, d_k)
    mask: tensor of shape (batch_size, 1, 1, seq_len)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output

# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, seq_len, d_model]
        bs = q.size(0)
        
        # Linear projections and split into num_heads
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        
        # Apply scaled dot-product attention
        scores = scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_k * self.num_heads)
        output = self.out_linear(concat)
        return output

# Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        # Self-attention
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        # Feed-forward
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        # Masked self-attention
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        # Encoder-decoder attention
        enc_dec_attn_output = self.enc_dec_attn(tgt, enc_output, enc_output, src_mask)
        tgt = self.norm2(tgt + self.dropout(enc_dec_attn_output))
        # Feed-forward
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        return tgt

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        # Pass through encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        # Embed and add positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        tgt = self.dropout(tgt)
        # Pass through decoder layers
        for layer in self.layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        return tgt

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)
    
    def make_src_mask(self, src, pad_idx):
        # src: [batch_size, src_seq_len]
        # mask: [batch_size, 1, 1, src_seq_len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt, pad_idx):
        # tgt: [batch_size, tgt_seq_len]
        # mask: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_seq_len]
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)
        return tgt_mask
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.out(dec_output)
        return output
