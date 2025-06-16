import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple
import os
import requests
import re
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length   
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]        
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[-self.max_length:]
        else:
            pad_id = self.tokenizer.pad_token_id
            token_ids = [pad_id] * (self.max_length - len(token_ids)) + token_ids        
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            'input_ids': input_ids,
            'labels': target_ids,
            'attention_mask': attention_mask
        }
def download_shakespeare_data(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    shakespeare_path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(shakespeare_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"     
        response = requests.get(url)
        with open(shakespeare_path, 'w', encoding='utf-8') as f:
            f.write(response.text)    
        print(f"Dataset downloaded to {shakespeare_path}")
    else:
        print(f"Dataset already exists at {shakespeare_path}")
    return shakespeare_path
def prepare_data(file_path: str, tokenizer, chunk_size: int = 512) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()    
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0    
    dialogue_pattern = re.compile(r'^[A-Z][A-Z\s]+:')  
    sentence_end = re.compile(r'[.!?]')
    scene_marker = re.compile(r'^SCENE|^ACT|^Enter|^Exit|^Exeunt', re.IGNORECASE)    
    current_speaker = None
    dialogue_context = []
    for line in lines:
        line = line.strip()
        if not line:
            continue            
        if scene_marker.match(line):
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_speaker = None
            dialogue_context = []
            continue            
        is_dialogue = bool(dialogue_pattern.match(line))        
        if is_dialogue:
            speaker = line.split(':')[0].strip()
            if speaker != current_speaker:
                current_speaker = speaker
                dialogue_context.append(f"{speaker} speaks:")        
        line_tokens = len(tokenizer.encode(line))
        if current_length + line_tokens > chunk_size - 20: 
            if current_chunk:
                last_sentence = current_chunk[-1]
                if not sentence_end.search(last_sentence):
                    for i in range(len(current_chunk) - 1, -1, -1):
                        if sentence_end.search(current_chunk[i]):
                            chunks.append(' '.join(current_chunk[:i+1]))
                            context_lines = current_chunk[i+1:]
                            current_chunk = context_lines
                            current_length = sum(len(tokenizer.encode(s)) for s in current_chunk)
                            break
                    else:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0        
        current_chunk.append(line)
        current_length += line_tokens        
        if is_dialogue and len(dialogue_context) > 3:
            dialogue_context.pop(0)    
    if current_chunk:
        chunks.append(' '.join(current_chunk))    
    processed_chunks = []
    for chunk in chunks:
        chunk = ' '.join(chunk.split())        
        if not any(chunk.startswith(s) for s in ['<BOS>', ' ']):
            if dialogue_pattern.match(chunk):
                chunk = '<BOS> ' + chunk
            else:
                context = ' '.join(dialogue_context[-2:]) if dialogue_context else ''
                chunk = f"<BOS> {context} {chunk}".strip()        
        if not any(chunk.endswith(p) for p in ['.', '!', '?']):
            chunk = chunk + '.'
        tokens = tokenizer.encode(chunk)
        if (20 <= len(tokens) <= chunk_size and
            len(re.findall(r'[.!?]', chunk)) >= 2 and 
            not all(c in '.,!? ' for c in chunk)): 
            processed_chunks.append(chunk)
    return processed_chunks
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def save_model(model: nn.Module, tokenizer, path: str, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__ if config else None,
        'vocab_size': tokenizer.get_vocab_size(),
    }
    torch.save(checkpoint, path)    
    tokenizer_dir = path.replace('.pt', '_tokenizer/')
    tokenizer.save(tokenizer_dir)
    print(f"Model saved to {path}")
    print(f"Tokenizer saved to {tokenizer_dir}")
def load_model(model_class, path: str, tokenizer):
    checkpoint = torch.load(path, map_location='cpu')    
    if checkpoint.get('config'):
        config_dict = checkpoint['config']
        model = model_class(
            vocab_size=checkpoint['vocab_size'],
            d_model=config_dict.get('d_model', 512),
            n_heads=config_dict.get('n_heads', 8),
            n_layers=config_dict.get('n_layers', 6),
            d_ff=config_dict.get('d_ff', 2048),
            max_seq_len=config_dict.get('max_seq_len', 512),
            dropout=config_dict.get('dropout', 0.1)
        )
    else:
        model = model_class(vocab_size=checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
class LearningRateScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr   
    def _get_lr(self):
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
def calculate_perplexity(loss: float) -> float:
    return torch.exp(torch.tensor(loss)).item()