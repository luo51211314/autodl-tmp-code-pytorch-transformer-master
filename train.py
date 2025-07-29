import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import time
import psutil
import torch.cuda as cuda

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk

import warnings
from tqdm import tqdm
from pathlib import Path

# nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore")


def log_metrics(epoch, train_loss, val_loss, bleu_score, log_file):
    # Get memory usage
    cpu_mem = psutil.virtual_memory().percent
    gpu_alloc = cuda.memory_allocated() / (1024 ** 2)  # MB
    gpu_cache = cuda.memory_reserved() / (1024 ** 2)  # MB

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    log_entry = (
        f"[{timestamp}] Epoch: {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"BLEU: {bleu_score:.2f} | "
        f"CPU Mem: {cpu_mem:.1f}% | "
        f"GPU Alloc: {gpu_alloc:.1f}MB | "
        f"GPU Cache: {gpu_cache:.1f}MB\n"
    )

    with open(log_file, 'a') as f:
        f.write(log_entry)

    print(log_entry.strip())


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input,
                                   torch.empty(1, 1).type_as(decoder_input).fill_(next_word.item()).to(device)], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt,
                   max_len, device, print_msg, global_state, writer, num_examples=-1):
    model.eval()
    count = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'))

    source_texts = []
    expected = []
    predicted = []
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation."

            # Calculate validation loss
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = criterion(proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                             label.view(-1))
            total_loss += loss.item()

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            if count < num_examples or num_examples == -1:
                print_msg('-' * console_width)
                print_msg(f'SOURCE: {source_text}')
                print_msg(f'TARGET: {target_text}')
                print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

    avg_loss = total_loss / count

    # Calculate BLEU score
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu([[ref] for ref in expected], predicted, smoothing_function=smoothie) * 100

    return avg_loss, bleu


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_from_disk('/root/autodl-tmp/datasets/opus_books')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 80% for training, 10% for validation and 10% for testing
    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size - val_ds_size

    # Split the dataset
    train_ds_raw, remaining = random_split(ds_raw, [train_ds_size, len(ds_raw) - train_ds_size])
    val_ds_raw, test_ds_raw = random_split(remaining, [val_ds_size, test_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt,
                               config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=8, shuffle=False)  # 适当增大batch_size

    return train_dataloader, val_dataloader, test_ds, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
                              src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'],
                              d_model=config['d_model'])
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_ds, tokenizer_src, tokenizer_tgt = get_ds(config)
    # Save test dataset for later use in test.py
    torch.save(test_ds, 'test_dataset.pt')

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preload model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]')).to(device)

    # Initialize logging
    with open(config['log_file'], 'w') as f:
        f.write("Starting training...\n")

    for epoch in range(initial_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        train_losses = []

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"Loss": f"Loss: {loss.item():6.3f}"})
            train_losses.append(loss.item())

            writer.add_scalar("train loss", loss.item(), global_step=global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Run validation and get BLEU score
        val_loss, bleu_score = run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt,
            config['seq_len'], device,
            lambda msg: batch_iterator.write(msg),
            global_step, writer,
            num_examples=2  # 仅展示2条，但计算全部
        )

        # Log metrics
        log_metrics(epoch, avg_train_loss, val_loss, bleu_score, config['log_file'])

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)