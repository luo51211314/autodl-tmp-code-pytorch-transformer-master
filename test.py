import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os
import datetime

import nltk
from tokenizers import Tokenizer
from datasets import load_from_disk
from itertools import islice

from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(decoder_input).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

def load_model(config, device):
    model = build_transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    model_filename = get_weights_file_path(config, config['preload'])
    print(f"Loading model weights from {model_filename}")
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    return model

def calculate_ppl(model, dataloader, tokenizer_tgt, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), reduction='none')
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating PPL"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            labels = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = criterion(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                labels.view(-1)
            )

            mask = (labels.view(-1) != tokenizer_tgt.token_to_id('[PAD]')).float()
            num_tokens = torch.sum(mask).item()
            total_loss += torch.sum(loss * mask).item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return ppl

def calculate_bleu(model, dataloader, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()
    hypotheses = []
    references = []
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating BLEU"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            model_out = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, max_len, device
            )

            model_out_text = tokenizer_tgt.decode(model_out.cpu().numpy())
            target_text = batch['tgt_text'][0]

            if not model_out_text.strip() or "[PAD]" in model_out_text:
                continue

            ref_tokens = nltk.word_tokenize(target_text.lower())
            hyp_tokens = nltk.word_tokenize(model_out_text.lower())

            references.append([ref_tokens])
            hypotheses.append(hyp_tokens)

    if not references:
        print("\nWarning: All predictions were empty or invalid! BLEU score set to 0.")
        return 0.0

    return corpus_bleu(references, hypotheses, smoothing_function=smoothie) * 100

def write_to_log(config, ppl, bleu):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "test_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - Epoch: {config['preload']} - PPL: {ppl:.2f} - BLEU: {bleu:.2f}\n")

def interactive_translate(model, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len, device):
    print("\nInteractive Translation Mode (type 'exit' to quit)")
    model.eval()
    pad_token_id = tokenizer_src.token_to_id("[PAD]")

    while True:
        source_text = input(f"\nEnter {src_lang} text: ")
        if source_text.lower() == 'exit':
            break

        enc_input_tokens = tokenizer_src.encode(source_text).ids
        enc_num_padding = seq_len - len(enc_input_tokens) - 2

        if enc_num_padding < 0:
            print(f"Input too long. Max length: {seq_len - 2}")
            continue

        encoder_input = torch.cat([
            torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
            torch.tensor([pad_token_id] * enc_num_padding, dtype=torch.int64)
        ]).unsqueeze(0).to(device)

        encoder_mask = (encoder_input != pad_token_id).unsqueeze(0).unsqueeze(0).int().to(device)

        model_out = greedy_decode(
            model, encoder_input, encoder_mask,
            tokenizer_src, tokenizer_tgt, seq_len, device
        )

        translation = tokenizer_tgt.decode(model_out.cpu().numpy())
        print(f"{tgt_lang} translation: {translation}")

def get_test_ds(config, tokenizer_src, tokenizer_tgt):
    ds_raw = load_from_disk('/root/autodl-tmp/datasets/opus_books')

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    _, test_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    test_ds = BilingualDataset(
        test_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )
    return DataLoader(test_ds, batch_size=1, shuffle=False)

if __name__ == '__main__':
    config = get_config()
    config['preload'] = '39'  # 或指定具体 epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    config['src_vocab_size'] = tokenizer_src.get_vocab_size()
    config['tgt_vocab_size'] = tokenizer_tgt.get_vocab_size()

    test_dataloader = get_test_ds(config, tokenizer_src, tokenizer_tgt)

    model = load_model(config, device)

    print("\nRunning evaluation...")
    ppl = calculate_ppl(model, test_dataloader, tokenizer_tgt, device)
    bleu_score = calculate_bleu(
        model, test_dataloader,
        tokenizer_src, tokenizer_tgt,
        config['seq_len'], device
    )

    print(f"\n{'='*40}")
    print(f"Evaluation Results for Epoch {config['preload']}:")
    print(f"- Perplexity (PPL): {ppl:.2f}")
    print(f"- BLEU Score: {bleu_score:.2f}")
    print(f"{'='*40}\n")

    write_to_log(config, ppl, bleu_score)

    interactive_translate(
        model, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'],
        config['seq_len'], device
    )