import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import re
from config import Config
from tokenizer import HFTokenizer
from model.transformer import GPTModel
from model.utils import (
    TextDataset, download_shakespeare_data, prepare_data,
    count_parameters, save_model, calculate_perplexity, LearningRateScheduler
)
class CoherenceLoss(nn.Module):
    def __init__(self, tokenizer, alpha=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.alpha = alpha
        self.tokenizer = tokenizer       
    def forward(self, logits, labels, input_ids):
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))        
        coherence_loss = 0.0        
        valid_mask = (labels != -100)
        if valid_mask.sum() > 0:
            last_tokens = input_ids[:, -1]
            proper_endings = torch.tensor([self.tokenizer.encode('.')[0], 
                                         self.tokenizer.encode('!')[0], 
                                         self.tokenizer.encode('?')[0]], 
                                        device=logits.device)
            ending_loss = 1.0 - torch.mean(torch.isin(last_tokens, proper_endings).float())
            coherence_loss += ending_loss
        char_name_pattern = r'^[A-Z][A-Z\s]+:'
        if re.search(char_name_pattern, self.tokenizer.decode(input_ids[0].tolist())):
            coherence_loss += 0.1  
        return ce_loss + self.alpha * coherence_loss
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, clip_grad_norm=0.5):
    model.train()
    total_loss, num_batches = 0.0, 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)        
        valid_mask = (labels != -100)
        if valid_mask.sum() == 0:
            continue            
        if isinstance(criterion, CoherenceLoss):
            loss = criterion(logits, labels, input_ids)
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches        
        if num_batches % 10 == 0:
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}', 
                'ppl': f'{calculate_perplexity(avg_loss):.2f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
    return total_loss / num_batches
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)            
            logits = model(input_ids, attention_mask)
            if hasattr(criterion, 'forward') and criterion.__class__.__name__ == 'CoherenceLoss':
                loss = criterion(logits, labels, input_ids)
            else:
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches
def generate_sample(model, tokenizer, prompt, config, max_length=40, temperature=0.45, top_k=15, top_p=0.65):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(config.device)
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(output_ids[0].tolist())        
        lines = generated_text.split('\n')
        processed_lines = []        
        for line in lines:
            if not line.strip():
                continue                
            if re.match(r'^[A-Z][A-Z\s]+:', line):
                char_name = line[:line.find(':')+1]
                dialogue = line[line.find(':')+1:].strip()                
                dialogue = re.sub(r'\s+', ' ', dialogue) 
                dialogue = re.sub(r'\s+([.,!?;:])', r'\1', dialogue)  
                if len(dialogue.split()) >= 3: 
                    processed_lines.append(f"{char_name} {dialogue}")
            else:
                if re.match(r'^[A-Z].*[.!?]$', line.strip()):
                    processed_lines.append(line.strip())        
        generated_text = '\n'.join(processed_lines)
        generated_text = re.sub(r'\n\s*\n', '\n', generated_text)  
        generated_text = generated_text.strip()
        return generated_text
def evaluate_generation_quality(model, tokenizer, config, test_prompts):
    print("\n=== Generation Quality Assessment ===")
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: '{prompt}'")        
        for temp in [0.3, 0.6, 0.8]:
            generated = generate_sample(model, tokenizer, prompt, config, max_length=40)
            print(f"Temp {temp}: {generated}")
        print("-" * 50)
def main():
    config = Config()
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("=== Training Transformer LLM from Scratch ===")
    print(f"Device: {config.device}")
    shakespeare_path = download_shakespeare_data()
    with open(shakespeare_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print("\n=== Training Tokenizer ===")
    tokenizer = HFTokenizer()
    tokenizer.train(shakespeare_path, vocab_size=config.vocab_size)
    tokenizer.save(config.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print("\n=== Preparing Dataset ===")
    text_chunks = prepare_data(shakespeare_path, tokenizer, config.max_seq_len)
    dataset = TextDataset(text_chunks, tokenizer, config.max_seq_len)
    train_size = int(0.85 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42)  
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print("\n=== Initializing Model ===")
    model = GPTModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(config.device)
    print(f"Model Parameters: {count_parameters(model):,}")
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = LearningRateScheduler(optimizer, config.d_model, config.warmup_steps)
    criterion = CoherenceLoss(tokenizer)
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 5  
    epochs_no_improve = 0
    best_model_path = os.path.join(config.model_dir, 'best_model.pt')
    train_losses = []
    val_losses = []
    for epoch in range(1, config.epochs + 1):
        print(f"\n=== Epoch {epoch}/{config.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, scheduler, config.gradient_clip)
        val_loss = validate_epoch(model, val_loader, criterion, config.device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {calculate_perplexity(val_loss):.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_model(model, tokenizer, best_model_path, config)
            print(f"[Checkpoint] Best model saved at epoch {epoch} with val loss {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break
    final_model_path = os.path.join(config.model_dir, 'final_model.pt')
    save_model(model, tokenizer, final_model_path, config)
    print(f"Final model saved to {final_model_path}")
if __name__ == "__main__":
    main()