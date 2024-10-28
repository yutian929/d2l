import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import re
project_root = "/home/yutian/projects/d2l/d2l"
sys.path.append(project_root)
# Import your dataset classes
from datasets.sequential.sentence_pair_preprocess import SentencePairDataset, TranslationDataset
from env.cuda import get_device
from transformer import Transformer


# After training the model, we can define an inference function
def translate_sentence(model, sentence, dataset, max_len=50, device='cpu'):
    model.eval()
    
    # Preprocess the input sentence
    sentence = dataset.sentence_preprocess(sentence)
    tokens = dataset.tokenize_sentence(sentence, lang='eng')
    
    # Convert tokens to indices
    src_indices = [dataset.vocab_eng.get(token, dataset.vocab_eng['<unk>']) for token in tokens]
    src_indices = [dataset.vocab_eng['<bos>']] + src_indices + [dataset.vocab_eng['<eos>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_seq_len]
    
    # Create source mask
    src_mask = model.make_src_mask(src_tensor, pad_idx=dataset.vocab_eng['<pad>'])
    
    # Encode the source sentence
    enc_src = model.encoder(src_tensor, src_mask)
    
    # Initialize the target sequence with <bos>
    tgt_indices = [dataset.vocab_chn['<bos>']]
    tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)  # [1, 1]
    
    # Initialize variables
    max_target_len = max_len
    idx_to_word_chn = {index: word for word, index in dataset.vocab_chn.items()}
    translated_tokens = []
    
    for _ in range(max_target_len):
        # Create target mask
        tgt_mask = model.make_tgt_mask(tgt_tensor, pad_idx=dataset.vocab_chn['<pad>'])
        
        # Decode
        output = model.decoder(tgt_tensor, enc_src, src_mask, tgt_mask)
        output = model.out(output)
        
        # Get the next token (greedy decoding)
        next_token = output[:, -1, :].argmax(1).item()
        
        # Append the predicted token to the target sequence
        tgt_indices.append(next_token)
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        
        # If the model predicts <eos>, stop decoding
        if next_token == dataset.vocab_chn['<eos>']:
            break
        
        # Collect the predicted word
        translated_tokens.append(idx_to_word_chn.get(next_token, '<unk>'))
    
    # Postprocess the translated tokens
    translated_sentence = ''.join(translated_tokens)
    translated_sentence = re.sub(r'(<eos>|<pad>|<unk>)', '', translated_sentence)
    translated_sentence = translated_sentence.strip()
    
    return translated_sentence


if __name__ == "__main__":
    # Hyperparameters
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0001
    max_len = 50

    # Instantiate the dataset and dataloader
    dataset = SentencePairDataset("/home/yutian/projects/d2l/d2l/datasets/sequential/cmn-eng/cmn.txt")
    translation_dataset = TranslationDataset(dataset, max_len=max_len)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True)

    # Device configuration
    device = get_device()

    # Instantiate the model
    src_vocab_size = len(dataset.vocab_eng)
    tgt_vocab_size = len(dataset.vocab_chn)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab_chn['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
            src_batch = src_batch.to(device)  # [batch_size, src_seq_len]
            tgt_batch = tgt_batch.to(device)  # [batch_size, tgt_seq_len]
            
            # Prepare masks
            src_mask = model.make_src_mask(src_batch, pad_idx=dataset.vocab_eng['<pad>'])
            tgt_mask = model.make_tgt_mask(tgt_batch[:, :-1], pad_idx=dataset.vocab_chn['<pad>'])
            
            # Shift target for input and output
            tgt_input = tgt_batch[:, :-1]  # Input to the decoder (without <eos>)
            tgt_output = tgt_batch[:, 1:]  # Target output (without <bos>)
            
            # Forward pass
            outputs = model(src_batch, tgt_input, src_mask, tgt_mask)
            outputs = outputs.reshape(-1, tgt_vocab_size)
            tgt_output = tgt_output.reshape(-1)
            
            # Compute loss
            loss = criterion(outputs, tgt_output)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Step [{batch_idx+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # # Save the model checkpoint
    # torch.save(model.state_dict(), 'transformer_model.pth')

    # Allow the user to input an English sentence
    while True:
        user_input = input("Enter an English sentence to translate (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        # Translate the sentence
        translation = translate_sentence(model, user_input, dataset, max_len=max_len, device=device)
        print(f"Translated Chinese Sentence: {translation}\n")
