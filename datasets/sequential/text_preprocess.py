import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import random

class TextDataset:
    def __init__(self, file_path, min_freq=2, seq_length=10, batch_size=64, use_random_iter=False):
        self.file_path = file_path
        self.min_freq = min_freq
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.use_random_iter = use_random_iter  # 添加 use_random_iter 参数

        # 执行数据处理流程
        self.text = self.read_text_file()
        self.tokens = self.tokenize_text()
        self.token_to_idx, self.idx_to_token = self.build_vocab()
        self.vocab_size = len(self.token_to_idx)
        self.indices = self.text_to_indices()
        # 根据采样方式创建数据集
        if self.use_random_iter:
            self.dataset = self.create_dataset_random()
        else:
            self.dataset = self.create_dataset_sequential()

        # 创建 DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, batch_sampler=None)

    # 1. 读取文本文件
    def read_text_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = self.preprocess_text(text)
        return text

    # 2. 数据预处理
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^A-Za-z]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # 3. 词元化
    def tokenize_text(self):
        tokens = self.text.split()
        return tokens

    # 4. 创建词汇表
    def build_vocab(self):
        token_counts = Counter(self.tokens)
        # 过滤掉出现频率低于 min_freq 的词元
        filtered_tokens = {token: count for token, count in token_counts.items() if count >= self.min_freq}

        # 添加特殊的 <unk> 词元
        idx_to_token = ['<unk>'] + sorted(filtered_tokens, key=filtered_tokens.get, reverse=True)
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

        return token_to_idx, idx_to_token

    # 5. 文本转换为索引序列
    def text_to_indices(self):
        indices = [self.token_to_idx.get(token, self.token_to_idx['<unk>']) for token in self.tokens]
        return indices

    # 6. 创建随机采样数据集
    def create_dataset_random(self):
        corpus = self.indices
        num_steps = self.seq_length
        batch_size = self.batch_size

        # 从随机偏移量开始对序列进行分区，随机范围包括 num_steps - 1
        offset = random.randint(0, num_steps - 1)
        corpus = corpus[offset:]
        # 减去1，是因为我们需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为 num_steps 的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 随机打乱初始索引
        random.shuffle(initial_indices)

        # 定义数据获取函数
        def data(pos):
            return corpus[pos: pos + num_steps]

        # 生成所有样本
        X = []
        Y = []
        for pos in initial_indices:
            X.append(data(pos))
            Y.append(data(pos + 1))
        # 转换为张量
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        # 创建 Dataset
        dataset = torch.utils.data.TensorDataset(X, Y)
        return dataset

    # 7. 创建顺序采样数据集
    def create_dataset_sequential(self):
        corpus = self.indices
        num_steps = self.seq_length
        batch_size = self.batch_size

        # 从随机偏移量开始划分序列
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs = Xs.reshape(batch_size, -1)
        Ys = Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps

        # 创建所有样本
        X = []
        Y = []
        for i in range(0, num_batches * num_steps, num_steps):
            X.append(Xs[:, i: i + num_steps])
            Y.append(Ys[:, i: i + num_steps])
        # 将列表中的张量拼接
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        # 创建 Dataset
        dataset = torch.utils.data.TensorDataset(X, Y)
        return dataset

    # 保存词汇表
    def save_vocab(self, token_to_idx_path='token_to_idx.pkl', idx_to_token_path='idx_to_token.pkl'):
        with open(token_to_idx_path, 'wb') as f:
            pickle.dump(self.token_to_idx, f)
        with open(idx_to_token_path, 'wb') as f:
            pickle.dump(self.idx_to_token, f)

    # 加载词汇表
    def load_vocab(self, token_to_idx_path='token_to_idx.pkl', idx_to_token_path='idx_to_token.pkl'):
        with open(token_to_idx_path, 'rb') as f:
            self.token_to_idx = pickle.load(f)
        with open(idx_to_token_path, 'rb') as f:
            self.idx_to_token = pickle.load(f)
        self.vocab_size = len(self.token_to_idx)
        
    # 画出词频图
    def plot_word_frequency(self):
        import matplotlib.pyplot as plt
        # 获取过滤后的词频列表
        token_counts = Counter(self.tokens)
        filtered_token_counts = {token: count for token, count in token_counts.items() if count >= self.min_freq}
        sorted_token_counts = sorted(filtered_token_counts.items(), key=lambda x: x[1], reverse=True)
        freqs = [freq for token, freq in sorted_token_counts]
        # 绘制词频分布图
        plt.figure(figsize=(8, 6))
        plt.loglog(range(1, len(freqs)+1), freqs)
        plt.xlabel('token:x')
        plt.ylabel('frequency:n(x)')
        plt.title('Word Frequency Distribution (Filtered)')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    file_path = '/home/yutian/projects/d2l/d2l/datasets/sequential/The Time Machine.txt'  # 请确保文件路径正确
    seq_length = 35
    batch_size = 32
    min_freq = 5
    use_random_iter = True  # 设置为 True 使用随机采样

    # 初始化 TextDataset 类
    text_dataset = TextDataset(file_path, min_freq=min_freq, seq_length=seq_length, batch_size=batch_size, use_random_iter=use_random_iter)

    # 可选：画出词频图
    # text_dataset.plot_word_frequency()

    # 获取 DataLoader
    dataloader = text_dataset.dataloader

    # 获取词汇表大小
    vocab_size = text_dataset.vocab_size
    print(f'词汇表大小：{vocab_size}')

    # 获取词汇映射
    token_to_idx = text_dataset.token_to_idx
    idx_to_token = text_dataset.idx_to_token

    # 保存词汇表（可选）
    # text_dataset.save_vocab()

    # 现在可以使用 dataloader 进行模型训练
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f'批次 {batch_idx}: 输入形状 {inputs.shape}, 目标形状 {targets.shape}')
        break  # 仅示例，实际训练时可去掉
