import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch

class SentencePairDataset:
    def __init__(self, source_text_path):
        self.source_text_path = source_text_path
        self.sentences = self.read_text_file()
        self.sentences = self.sentences_preprocess_common(self.sentences)
        self.sentences_eng, self.sentences_chn = self.sentences_preprocess_special(self.sentences)
        
        # 分词
        self.tokenized_eng = [self.tokenize_sentence(s, lang='eng') for s in self.sentences_eng]
        self.tokenized_chn = [self.tokenize_sentence(s, lang='chn') for s in self.sentences_chn]
        
        # 创建词汇表
        self.vocab_eng = self.build_vocab(self.tokenized_eng, reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
        self.vocab_chn = self.build_vocab(self.tokenized_chn, reserved_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])
        
    def tokenize_sentence(self, sentence, lang='eng'):
        """根据语言类型对句子分词，英文用空格分词，中文字符之间加空格"""
        if lang == 'eng':
            return sentence.split()
        elif lang == 'chn':
            return list(sentence.replace(" ", ""))  # 每个中文字符单独分词，去除多余空格
    
    def build_vocab(self, tokenized_sentences, reserved_tokens):
        """创建词汇表并返回token到索引的映射"""
        counter = Counter(token for sentence in tokenized_sentences for token in sentence)
        vocab = {token: i for i, token in enumerate(reserved_tokens)}  # 加入保留的token
        for token, freq in counter.items():
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab
    
    def sentences_preprocess_common(self, sentences):
        return [s.lower() for s in sentences]
    
    def sentences_preprocess_special(self, sentences):
        sentences_english, sentences_chinese = [], []
        for s in sentences:
            s_eng, s_chn = s.split("\t")[:2]
            sentences_english.append(self.sentence_preprocess(s_eng))
            sentences_chinese.append(self.sentence_preprocess(s_chn))
        return sentences_english, sentences_chinese
    
    def sentence_preprocess(self, sentence):
        sentence = sentence.replace('\u202f', ' ').replace('\xa0', ' ')
        sentence = re.sub(r'([.!?,"():;。，！？：；（）“”‘’])', r' \1 ', sentence)
        sentence = re.sub(r'([\u4e00-\u9fff])', r' \1 ', sentence)
        sentence = re.sub(r'\s{2,}', ' ', sentence).strip()
        return sentence
    
    def read_text_file(self):
        with open(self.source_text_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        return sentences


class TranslationDataset(Dataset):
    def __init__(self, dataset, max_len=50):
        self.dataset = dataset
        self.max_len = max_len
        self.pad_id_eng = self.dataset.vocab_eng['<pad>']
        self.pad_id_chn = self.dataset.vocab_chn['<pad>']
        
    def __len__(self):
        return len(self.dataset.tokenized_eng)
    
    def __getitem__(self, idx):
        src_sentence = self.dataset.tokenized_eng[idx]
        tgt_sentence = self.dataset.tokenized_chn[idx]
        
        # 将每个句子加上<bos>和<eos>标记，并映射到索引
        src_ids = [self.dataset.vocab_eng['<bos>']] + [self.dataset.vocab_eng.get(token, self.dataset.vocab_eng['<unk>']) for token in src_sentence] + [self.dataset.vocab_eng['<eos>']]
        tgt_ids = [self.dataset.vocab_chn['<bos>']] + [self.dataset.vocab_chn.get(token, self.dataset.vocab_chn['<unk>']) for token in tgt_sentence] + [self.dataset.vocab_chn['<eos>']]
        
        # 填充到最大长度
        src_ids = src_ids[:self.max_len] + [self.pad_id_eng] * max(0, self.max_len - len(src_ids))
        tgt_ids = tgt_ids[:self.max_len] + [self.pad_id_chn] * max(0, self.max_len - len(tgt_ids))
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


if __name__ == "__main__":
    # 实例化数据集
    max_len = 50  # 假设的最大句子长度
    dataset = SentencePairDataset("/home/yutian/projects/d2l/d2l/datasets/sequential/cmn-eng/cmn.txt")
    translation_dataset = TranslationDataset(dataset, max_len=max_len)

    # 添加idx到词汇的映射，用于索引还原成词汇
    def index_to_word(vocab):
        return {index: word for word, index in vocab.items()}

    # 获取vocab的逆映射
    idx_to_word_eng = index_to_word(dataset.vocab_eng)
    idx_to_word_chn = index_to_word(dataset.vocab_chn)

    # 定义一个函数用于打印单个句子的索引和对应的字符
    def print_sentence(indices, idx_to_word):
        words = [idx_to_word[idx.item()] for idx in indices if idx.item() in idx_to_word]
        print("Indices:", indices.tolist())
        print("Words:", " ".join(words))

    # 创建DataLoader
    batch_size = 2  # 设置为小批量以便观察
    dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True)

    # 获取一个batch进行打印
    for src_batch, tgt_batch in dataloader:
        print("---- English Sentences (Source) ----")
        for sentence_indices in src_batch:
            print_sentence(sentence_indices, idx_to_word_eng)
        
        print("\n---- Chinese Sentences (Target) ----")
        for sentence_indices in tgt_batch:
            print_sentence(sentence_indices, idx_to_word_chn)
        
        break  # 仅打印一个batch的内容

# ---- English Sentences (Source) ----
# Indices: [1, 720, 74, 302, 685, 158, 399, 847, 20, 12, 262, 595, 591, 228, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Words: <bos> could you please talk a bit louder ? i can't hear very well . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
# Indices: [1, 302, 45, 158, 617, 413, 134, 1819, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Words: <bos> please call a taxi for this lady . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

# ---- Chinese Sentences (Target) ----
# Indices: [1, 6, 51, 437, 677, 69, 1324, 35, 33, 19, 48, 23, 227, 126, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Words: <bos> 你 能 大 声 点 讲 吗 ？ 我 听 不 太 清 。 <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
# Indices: [1, 95, 58, 199, 1081, 1030, 466, 210, 594, 41, 626, 588, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Words: <bos> 请 为 这 位 女 士 叫 辆 出 租 车 。 <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
