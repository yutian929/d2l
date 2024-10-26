import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from seq2seq import Seq2Seq, Encoder, Decoder

import sys
import os
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)

# Import your dataset classes
from datasets.sequential.sentence_pair_preprocess import SentencePairDataset, TranslationDataset
from env.cuda import get_device

def index_to_word(vocab):
    return {index: word for word, index in vocab.items()}

def print_sentence(indices, idx_to_word):
    words = []
    for idx in indices:
        idx = idx.item()
        word = idx_to_word.get(idx, '<unk>')
        words.append(word)
    print("Words:", " ".join(words))

def train_model(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for src_batch, trg_batch in dataloader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)
        
        optimizer.zero_grad()
        
        output = model(src_batch, trg_batch)
        # output: [batch_size, trg_len, output_dim]
        # trg_batch: [batch_size, trg_len]
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # exclude <bos> token
        trg = trg_batch[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Clip gradients to prevent exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src_batch, trg_batch in dataloader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)
            
            output = model(src_batch, trg_batch, teacher_forcing_ratio=0)
            # output: [batch_size, trg_len, output_dim]
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # exclude <bos> token
            trg = trg_batch[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def preprocess_sentence(sentence, dataset, max_len=50):
    # Lowercase and preprocess the sentence
    sentence = sentence.lower()
    sentence = dataset.sentence_preprocess(sentence)
    # Tokenize
    tokens = dataset.tokenize_sentence(sentence, lang='eng')
    # Add <bos> and <eos> tokens
    tokens = ['<bos>'] + tokens + ['<eos>']
    # Convert tokens to indices
    indices = [dataset.vocab_eng.get(token, dataset.vocab_eng['<unk>']) for token in tokens]
    # Pad or truncate to max_len
    indices = indices[:max_len] + [dataset.vocab_eng['<pad>']] * max(0, max_len - len(indices))
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(indices).unsqueeze(0)  # Shape: [1, max_len]
    return tensor

def translate_sentence(model, sentence_tensor, dataset, max_len=50, device='cpu'):
    model.eval()
    sentence_tensor = sentence_tensor.to(device)
    
    # Encode the source sentence
    with torch.no_grad():
        encoder_hidden = model.encoder(sentence_tensor)
    
    # Create a tensor to hold the output indices
    trg_indices = [dataset.vocab_chn['<bos>']]
    
    # Initialize the decoder input and hidden state
    decoder_input = torch.tensor([dataset.vocab_chn['<bos>']], device=device)
    decoder_hidden = encoder_hidden
    
    for _ in range(max_len):
        with torch.no_grad():
            output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # Get the highest predicted token from output
            top1 = output.argmax(1)
            # Append prediction to output indices
            trg_indices.append(top1.item())
            # Break if <eos> token is generated
            if top1.item() == dataset.vocab_chn['<eos>']:
                break
            # Next input is the predicted token
            decoder_input = top1
        
    return trg_indices

def decode_translation(trg_indices, idx_to_word_chn):
    words = [idx_to_word_chn.get(idx, '<unk>') for idx in trg_indices]
    # Remove the <bos> token
    words = words[1:]
    # Stop at the first <eos> token
    if '<eos>' in words:
        words = words[:words.index('<eos>')]
    translation = ''.join(words)  # For Chinese, joining without spaces
    return translation

def main():
    # Hyperparameters
    INPUT_DIM = None  # to be set after loading vocab
    OUTPUT_DIM = None  # to be set after loading vocab
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    BATCH_SIZE = 64
    N_EPOCHS = 20
    CLIP = 1
    MAX_LEN = 50
    DEVICE = get_device()
    
    # Load dataset
    dataset = SentencePairDataset("/home/yutian/projects/d2l/d2l/datasets/sequential/cmn-eng/cmn.txt")
    translation_dataset = TranslationDataset(dataset, max_len=MAX_LEN)
    
    # Update input and output dimensions
    INPUT_DIM = len(dataset.vocab_eng)
    OUTPUT_DIM = len(dataset.vocab_chn)
    PAD_IDX_ENG = dataset.vocab_eng['<pad>']
    PAD_IDX_CHN = dataset.vocab_chn['<pad>']
    
    # DataLoader
    dataloader = DataLoader(translation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize encoder, decoder, and Seq2Seq model
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, PAD_IDX_ENG)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, PAD_IDX_CHN)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_CHN)
    
    # Training loop
    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, dataloader, optimizer, criterion, CLIP, DEVICE)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
    
    # Save the model
    # torch.save(model.state_dict(), 'seq2seq_model.pth')

    # Load the reverse mapping for Chinese vocabulary
    idx_to_word_chn = index_to_word(dataset.vocab_chn)
    
    print("\nTranslation testing:")
    # Allow the user to input sentences for translation
    while True:
        sentence = input("Enter an English sentence (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        # Preprocess the input sentence
        sentence_tensor = preprocess_sentence(sentence, dataset, max_len=MAX_LEN)
        # Translate the sentence
        trg_indices = translate_sentence(model, sentence_tensor, dataset, max_len=MAX_LEN, device=DEVICE)
        # Decode the translation
        translation = decode_translation(trg_indices, idx_to_word_chn)
        print(f"Translated Sentence: {translation}\n")
    
if __name__ == "__main__":
    main()
