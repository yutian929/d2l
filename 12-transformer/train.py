import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import sys
import os
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)

# Import your dataset classes
from datasets.sequential.sentence_pair_preprocess import SentencePairDataset, TranslationDataset
from env.cuda import get_device
from transfomer import TransformerModel

# 生成掩码函数
def create_mask(src, tgt, PAD_IDX_SRC, PAD_IDX_TGT):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # 创建源序列的填充掩码
    src_padding_mask = (src == PAD_IDX_SRC)  # [batch_size, src_seq_len]

    # 创建目标序列的填充掩码
    tgt_padding_mask = (tgt == PAD_IDX_TGT)  # [batch_size, tgt_seq_len]

    # 创建目标序列的序列掩码，防止模型看到未来的信息
    tgt_seq_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)  # [tgt_seq_len, tgt_seq_len]

    return src_padding_mask, tgt_padding_mask, tgt_seq_mask


# 训练循环
def train_epoch(model, dataloader, optimizer, criterion, PAD_IDX_SRC, PAD_IDX_TGT):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src = src.to(model.src_embedding.weight.device)
        tgt = tgt.to(model.src_embedding.weight.device)

        tgt_input = tgt[:, :-1]  # 移除句子末尾的一个词
        tgt_output = tgt[:, 1:]  # 移除句子开始的一个词

        src_padding_mask, tgt_padding_mask, tgt_seq_mask = create_mask(src, tgt_input, PAD_IDX_SRC, PAD_IDX_TGT)

        optimizer.zero_grad()

        # 前向传播
        output = model(
            src=src,
            tgt=tgt_input,
            tgt_mask=tgt_seq_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )  # 输出形状: [batch_size, tgt_seq_len - 1, tgt_vocab_size]

        # 将输出和目标展平成二维张量，以计算损失
        output = output.reshape(-1, output.shape[-1])  # [batch_size * (tgt_seq_len - 1), tgt_vocab_size]
        tgt_output = tgt_output.reshape(-1)  # [batch_size * (tgt_seq_len - 1)]

        loss = criterion(output, tgt_output)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# 翻译函数
def translate(model, sentence, dataset, max_len=50, device='cpu'):
    model.eval()

    src = preprocess_sentence(sentence, dataset, max_len=max_len, lang='eng')  # [1, seq_len]
    src = src.to(device)
    src_padding_mask = (src == dataset.vocab_eng['<pad>']).to(device)

    memory = model.encode(src, src_padding_mask)

    tgt_indices = [dataset.vocab_chn['<bos>']]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)  # [1, tgt_seq_len]
        tgt_padding_mask = (tgt_tensor == dataset.vocab_chn['<pad>']).to(device)
        tgt_seq_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt_indices)).to(device)

        output = model.decode(
            tgt=tgt_tensor,
            memory=memory,
            tgt_mask=tgt_seq_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = model.generator(output.transpose(0, 1))  # [1, tgt_seq_len, tgt_vocab_size]

        next_token = output.argmax(-1)[:, -1].item()
        tgt_indices.append(next_token)

        if next_token == dataset.vocab_chn['<eos>']:
            break

    tgt_vocab_inv = {idx: word for word, idx in dataset.vocab_chn.items()}
    translation = [tgt_vocab_inv.get(idx, '<unk>') for idx in tgt_indices[1:-1]]  # 去掉<bos>和<eos>

    return ''.join(translation)


# 预处理输入句子
def preprocess_sentence(sentence, dataset, max_len=50, lang='eng'):
    # 句子预处理
    sentence = sentence.lower()
    sentence = dataset.sentence_preprocess(sentence)
    tokens = dataset.tokenize_sentence(sentence, lang=lang)

    # 添加<bos>和<eos>
    tokens = ['<bos>'] + tokens + ['<eos>']
    # 将词转换为索引
    if lang == 'eng':
        indices = [dataset.vocab_eng.get(token, dataset.vocab_eng['<unk>']) for token in tokens]
        pad_idx = dataset.vocab_eng['<pad>']
    else:
        indices = [dataset.vocab_chn.get(token, dataset.vocab_chn['<unk>']) for token in tokens]
        pad_idx = dataset.vocab_chn['<pad>']

    # 填充或截断
    indices = indices[:max_len] + [pad_idx] * max(0, max_len - len(indices))

    return torch.tensor(indices).unsqueeze(0)  # [1, seq_len]


def main():
    # 设置设备
    device = get_device()

    # 实例化数据集和数据加载器
    max_len = 50
    batch_size = 64
    dataset = SentencePairDataset("/home/yutian/projects/d2l/d2l/datasets/sequential/cmn-eng/cmn.txt")
    translation_dataset = TranslationDataset(dataset, max_len=max_len)
    dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True)

    # 获取词汇表大小和特殊标记索引
    src_vocab_size = len(dataset.vocab_eng)
    tgt_vocab_size = len(dataset.vocab_chn)
    PAD_IDX_SRC = dataset.vocab_eng['<pad>']
    PAD_IDX_TGT = dataset.vocab_chn['<pad>']

    # 实例化模型、损失函数和优化器
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_TGT)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    num_epochs = 10 
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, PAD_IDX_SRC, PAD_IDX_TGT)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'transformer_translation_model.pth')

    # 测试模型
    model.load_state_dict(torch.load('transformer_translation_model.pth'))
    model.to(device)
    model.eval()

    while True:
        sentence = input("请输入英文句子（输入'quit'退出）：")
        if sentence.lower() == 'quit':
            break
        translation = translate(model, sentence, dataset, max_len=max_len, device=device)
        print(f"翻译结果：{translation}")


if __name__ == "__main__":
    main()
