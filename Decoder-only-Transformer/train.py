from datasets.table import list_table_cache_files
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

from pathlib import Path

def get_all_sentences(ds):
  for item in ds:
    yield item['text']


def get_or_build_tokenizer(config, ds):
  tokenizer_path = Path(config['tokenizer_file'])
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[EOS]'], min_frequency = 2)
    tokenizer.train_from_iterator(get_all_sentences(ds), trainer = trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer


def greedy_decode(model, decoder_input, tokenizer, max_len, device):
    eos_idx = tokenizer.token_to_id('[EOS]')

    while True:
        if decoder_input.size(1) +25 == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(decoder_input).to(device)

        # calculate output
        out = model.decode(decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(decoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            
            decoder_input = batch["input_ids"]
            if decoder_input.size(-1)<25:
              continue
             
            decoder_input = decoder_input[:, :torch.randint(12, 24, size=(1,)).item()].to(device) # (b, seq_len)
            prompt = tokenizer.decode(decoder_input.squeeze().detach().cpu().numpy())

            # check that the batch size is 1
            assert decoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, decoder_input, tokenizer, max_len, device)

            labels = batch["labels"][0]
            labels = tokenizer.decode(labels.squeeze().numpy())
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'PROMPT: ':>12}{prompt}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            print_msg(f"{f'ACTUAL: ':>12}{labels}")
            

            if count == num_examples:
                print_msg('-'*console_width)
                break


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def group_texts(examples, seq_len=350):
    concatenated_ids = sum(examples["input_ids"], [])
    total_length = len(concatenated_ids)
    
    # Drop the small trailing remainder
    total_length = (total_length // seq_len) * seq_len

    result = {
        "input_ids": [
            concatenated_ids[i : i + seq_len] for i in range(0, total_length, seq_len)
        ]
    }

    result["labels"] = result["input_ids"].copy()
    return result

def get_ds(config):
    ds_raw = load_dataset("roneneldan/TinyStories", split="train+validation")
    
    tokenizer = get_or_build_tokenizer(config, ds_raw)
    seq_len = config["seq_len"]

    # 1. Map the tokenization and grouping on the FULL dataset first
    def tokenize_function(examples):
        return {"input_ids": [tokenizer.encode(text).ids + [tokenizer.token_to_id("[EOS]")] for text in examples["text"]]}

    tokenized_ds = ds_raw.map(
        tokenize_function, batched=True, remove_columns=ds_raw.column_names
    )
    
    chunked_ds = tokenized_ds.map(lambda ex: group_texts(ex, seq_len), batched=True)

    # 2. Now split the preprocessed Hugging Face dataset into train and validation
    train_ds_size = int(0.9 * len(chunked_ds))
    val_ds_size = len(chunked_ds) - train_ds_size
    train_subset, val_subset = random_split(chunked_ds, [train_ds_size, val_ds_size])

    # 3. Wrap them in your custom BilingualDataset
    train_dataset = BilingualDataset(train_subset, seq_len)
    val_dataset = BilingualDataset(val_subset, seq_len)

    print(f"[Info] Total number of train samples: {len(train_dataset)}")
    print(f"[Info] Total number of validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_len ):
  model = build_transformer(vocab_len, config['seq_len'], d_model = config['d_model'])
  return model



def train_model(config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info] Using device: {device}")

  Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

  train_dataloader, val_dataloader, tokenizer = get_ds(config)
  model = get_model(config, tokenizer.get_vocab_size(), ).to(device)

  # Tensorboard
  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

  initial_epoch = 0
  global_step = 0
  if config['preload']:

    model_filename = get_weights_file_path(config, config['preload'])
    if Path.exists(Path(model_filename)):
      print(f"[Info] Preloading model {model_filename}")
      state = torch.load(model_filename)
      model.load_state_dict(state['model_state_dict'])
      initial_epoch = state['epoch'] + 1
      optimizer.load_state_dict(state['optimizer_state_dict'])
      global_step = state['global_step']

  loss_fn = nn.CrossEntropyLoss(label_smoothing = 0.1).to(device)

  for epoch in range(initial_epoch, config['num_epochs']):

    batch_iterator = tqdm(train_dataloader, desc = f"processing Epoch {epoch + 1 :02d}")
    for batch in batch_iterator:
      model.train()
      optimizer.zero_grad()
      decoder_input = batch['input_ids'].to(device)
      decoder_mask = batch['mask'].to(device)
      decoder_labels = batch['labels'].to(device)

      
      decoder_output = model.decode(decoder_input, decoder_mask)
      proj_output = model.project(decoder_output)

      #label = batch['label'].to(device)
      loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), decoder_labels.view(-1))
      batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

      writer.add_scalar('train_loss', loss.item(), global_step)
      writer.flush()

      loss.backward()
      optimizer.step()
      global_step += 1
    
    run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  config = get_config()
  train_model(config)