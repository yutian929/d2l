import re


class SentencePairDataset:
    def __init__(self, source_text_path):
        self.source_text_path = source_text_path
        self.vocab_eng, self.vocab_chn = self.build_vocab()

    def build_vocab(self):
        sentences = self.read_text_file()
        sentences = self.sentences_preprocess_common(sentences)
        sentences_eng, sentences_chn = self.sentences_preprocess_special(sentences)
        for i in range(1, 11):
            print(f"{-i}: {sentences_eng[-i]}\t{sentences_chn[-i]}")



    def sentences_preprocess_common(self, sentences):
        sentences = [s.lower() for s in sentences]
        return sentences
    
    def sentences_preprocess_special(self, sentences):
        # 针对d2l/datasets/sequential/cmn-eng/cmn.txt
        # Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
        # 分离英文和中文，删除后续的东西
        sentences_english = []
        sentences_chinese = []
        for s in sentences:
            s_eng, s_chn = s.split("\t")[:2]
            s_eng = self.sentence_preprocess(s_eng)
            s_chn = self.sentence_preprocess(s_chn)
            sentences_english.append(s_eng)
            sentences_chinese.append(s_chn)
        return sentences_english, sentences_chinese
    
    def sentence_preprocess(self, sentence):
        # 针对分离出来的原始的eng和chn句子
        sentence = sentence.replace('\u202f', ' ').replace('\xa0', ' ')  # 使用空格替换不间断空格
        sentence = re.sub(r'([.!?,"():;。，！？：；（）“”‘’])', r' \1 ', sentence)  # 在标点符号前后添加空格
        sentence = re.sub(r'([\u4e00-\u9fff])', r' \1 ', sentence)  # 在中文字符前后添加空格
        sentence = re.sub(r'\s{2,}', ' ', sentence).strip()  # 替换多个空格为单个空格，确保不会出现多余空格
        return sentence

    def read_text_file(self):
        with open(self.source_text_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        return sentences


if __name__ == "__main__":
    dataset = SentencePairDataset("/mnt/zhangyutian/projects/task10_yutian/d2l/d2l/datasets/sequential/cmn-eng/cmn.txt")
