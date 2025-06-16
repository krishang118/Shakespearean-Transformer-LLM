import torch
import argparse
import re
from config import Config
from tokenizer import HFTokenizer
from model.transformer import GPTModel
from model.utils import load_model
def optimize_parameters(prompt: str) -> dict:
    params = {
        'max_length': 100,  
        'temperature': 0.4,  
        'top_k': 30,       
        'top_p': 0.6,      
        'repetition_penalty': 1.3,  
        'beam_size': 4    
    }    
    prompt_length = len(prompt.split())
    if prompt_length > 10:
        params['max_length'] = 150  
        params['temperature'] = 0.5 
    elif prompt_length < 5:
        params['max_length'] = 80   
        params['temperature'] = 0.3 
    return params
def clean_generated_text(text):
    text = text.replace('<BOS>', '').replace('<EOS>', '')    
    text = text.replace(' : ', ' ').replace(': ', '')    
    text = ' '.join(text.split())
    text = text.replace(' .', '.').replace(' ,', ',')
    text = text.replace(' !', '!').replace(' ?', '?')
    text = text.replace(' :', ':').replace(': ', ':')    
    text = text.replace('?', '? ').replace('!', '! ').replace('.', '. ')
    text = ' '.join(text.split())     
    prefixes = ['KING', 'QUEEN', 'PRINCE', 'DUKE', 'DUCHESS', 'LORD', 'LADY', 'SIR', 'MISTRESS', 'CAPTAIN', 'GENTLEMAN', 'MESSENGER', 'SERVANT', 'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'SIXTH', 'SEVENTH', 'EIGHTH', 'NINTH', 'TENTH']
    prefix_pattern = r'|'.join(prefixes)
    pattern = re.compile(rf'((?:{prefix_pattern})?[A-Z][A-Z\s]+)([A-Z][a-z])')
    text = pattern.sub(lambda m: m.group(1).strip() + ': ' + m.group(2), text)
    text = re.sub(r'([A-Z][A-Z]+)([A-Z][a-z])', r'\1: \2', text)    
    sentences = text.split('. ')    
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith(('.', '!', '?', ':')):
            sentence += '.'
        processed_sentences.append(sentence)    
    return ' '.join(processed_sentences)
def generate_text(model, tokenizer, prompt, max_length=150, device=None):
    prompt_length = len(prompt.split())
    if prompt_length <= 5:  
        max_length = 50
    elif prompt_length <= 10: 
        max_length = 100
    temperature = 0.7 
    top_p = 0.9     
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    if device is not None:
        input_ids = input_ids.to(device)    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )    
    generated_text = tokenizer.decode(output[0].tolist())    
    first_idx = generated_text.find(prompt)
    if first_idx == -1:
        generated_text = prompt + " " + generated_text
        first_idx = 0    
    generated_text = generated_text[first_idx:]    
    generated_text = generated_text.replace(prompt, '', 1)    
    generated_text = prompt + generated_text    
    cleaned_text = clean_generated_text(generated_text)    
    if not cleaned_text.startswith(prompt):
        cleaned_text = prompt + cleaned_text[len(prompt):]    
    lines = cleaned_text.split('\n')
    processed_lines = []
    current_speaker = None
    for line in lines:
        line = line.strip()
        if not line:
            continue            
        if re.match(r'^[A-Z][A-Z\s]+:', line):
            char_name = line[:line.find(':')+1]
            dialogue = line[line.find(':')+1:].strip()
            if len(dialogue.split()) >= 3: 
                if any(word in dialogue.lower() for word in ['the', 'and', 'but', 'or', 'if', 'when', 'why', 'how']):
                    processed_lines.append(f"{char_name} {dialogue}")
                    current_speaker = char_name
        else:
            if re.match(r'^[A-Z].*[.!?]$', line):
                processed_lines.append(line)
    cleaned_text = '\n'.join(processed_lines)
    return cleaned_text.strip()
def advanced_generate(model, input_ids, max_length, tokenizer, temperature=0.7, 
                     top_k=None, top_p=None, repetition_penalty=1.1):
    generated = input_ids.clone()
    past_tokens = input_ids.clone()    
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature            
            if repetition_penalty != 1.0:
                for token_id in past_tokens.flatten():
                    next_token_logits[0, token_id] /= repetition_penalty            
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1:]
                next_token_logits[indices_to_remove] = float('-inf')            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == tokenizer.tokenizer.token_to_id("<EOS>"):
                break
            generated = torch.cat([generated, next_token], dim=1)
            past_tokens = torch.cat([past_tokens, next_token], dim=1)            
            if generated.size(1) >= model.max_seq_len:
                break
    return generated
def greedy_generate(model, input_ids, max_length, tokenizer):
    generated = input_ids.clone()
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token.item() == tokenizer.tokenizer.token_to_id("<EOS>"):
                break
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) >= model.max_seq_len:
                break
    return generated
def beam_search_generate(model, input_ids, max_length, tokenizer, beam_size=4):
    batch_size = input_ids.size(0)
    vocab_size = tokenizer.get_vocab_size()
    beams = [(input_ids, 0.0, 0)] 
    for _ in range(max_length):
        new_beams = []
        for seq, score, length in beams:
            if seq.size(1) >= model.max_seq_len:
                new_beams.append((seq, score, length))
                continue
            with torch.no_grad():
                logits = model(seq)
                next_token_logits = logits[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)    
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                for i in range(beam_size):
                    token_id = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    token_score = top_log_probs[0, i].item()
                    new_length = length + 1
                    length_penalty = ((5 + new_length) / 6) ** 0.8  
                    normalized_score = (score + token_score) / length_penalty
                    if token_id.item() == tokenizer.tokenizer.token_to_id("<EOS>"):
                        new_beams.append((seq, normalized_score, new_length))
                    else:
                        new_seq = torch.cat([seq, token_id], dim=1)
                        new_beams.append((new_seq, normalized_score, new_length))        
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]        
        if all(beam[0].size(1) >= model.max_seq_len or 
               beam[0][0, -1].item() == tokenizer.tokenizer.token_to_id("<EOS>") 
               for beam in beams):
            break    
    return beams[0][0]
def post_process_text(text, prompt=None):
    text = re.sub(r'<BOS>|<EOS>|<PAD>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def interactive_generation(model, tokenizer, config):
    print("=== Interactive Text Generation ===")
    print("Type 'quit' to exit, 'settings' to change parameters")
    print("Commands: 'compare' - compare different strategies")
    print("         'examples' - see example prompts")    
    settings = {
        'max_length': 80,
        'temperature': 0.6,
        'top_k': 20,
        'top_p': 0.8,
        'strategy': 'nucleus',
        'repetition_penalty': 1.1
    }
    print("\nCurrent settings:")
    print_settings(settings)
    while True:
        prompt = input("\nEnter your prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'settings':
            settings = update_settings(settings)
            continue
        elif prompt.lower() == 'compare':
            compare_strategies(model, tokenizer, config, settings)
            continue
        elif prompt.lower() == 'examples':
            show_examples()
            continue
        elif not prompt:
            continue
        try:
            generate_text(model, tokenizer, prompt, settings['max_length'], device=config.device)
        except Exception as e:
            print(f"Error during generation: {e}")
def compare_strategies(model, tokenizer, config, base_settings):
    prompt = input("Enter prompt for comparison: ").strip()
    if not prompt:
        return
    strategies = [
        ('greedy', {}),
        ('nucleus', {'temperature': 0.5, 'top_p': 0.8}),
        ('nucleus', {'temperature': 0.7, 'top_p': 0.9}),
        ('top_k', {'temperature': 0.6, 'top_k': 15}),
        ('beam', {})
    ]
    print(f"\n=== Comparing Strategies for: '{prompt}' ===")
    for strategy, params in strategies:
        settings = base_settings.copy()
        settings.update(params)
        settings['strategy'] = strategy
        settings['max_length'] = 60  
        print(f"\n--- {strategy.upper()} ---")
        try:
            generate_text(model, tokenizer, prompt, settings['max_length'], device=config.device)
        except Exception as e:
            print(f"Error with {strategy}: {e}")
def show_examples():
    examples = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo and Juliet",
        "In fair Verona",
        "But soft, what light",
        "A rose by any other name",
        "The course of true love",
        "Now is the winter"
    ]
    print("\n=== Example Prompts ===")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    print()
def print_settings(settings):
    print(f"  Strategy: {settings['strategy']}")
    print(f"  Max length: {settings['max_length']}")
    print(f"  Temperature: {settings['temperature']}")
    print(f"  Top-k: {settings['top_k']}")
    print(f"  Top-p: {settings['top_p']}")
    print(f"  Repetition penalty: {settings['repetition_penalty']}")
def update_settings(settings):
    print("\n=== Update Settings ===")
    print("Press Enter to keep current value")
    strategy = input(f"Strategy (greedy/top_k/nucleus/beam) [{settings['strategy']}]: ").strip()
    if strategy and strategy in ['greedy', 'top_k', 'nucleus', 'beam']:
        settings['strategy'] = strategy    
    max_length = input(f"Max length [{settings['max_length']}]: ").strip()
    if max_length and max_length.isdigit():
        settings['max_length'] = int(max_length)
    temperature = input(f"Temperature [{settings['temperature']}]: ").strip()
    if temperature:
        try:
            temp_val = float(temperature)
            if 0.1 <= temp_val <= 2.0:
                settings['temperature'] = temp_val
            else:
                print("Temperature should be between 0.1 and 2.0")
        except ValueError:
            print("Invalid temperature, keeping current value")
    top_k = input(f"Top-k [{settings['top_k']}]: ").strip()
    if top_k and top_k.isdigit():
        settings['top_k'] = int(top_k)
    top_p = input(f"Top-p [{settings['top_p']}]: ").strip()
    if top_p:
        try:
            p_val = float(top_p)
            if 0.1 <= p_val <= 1.0:
                settings['top_p'] = p_val
            else:
                print("Top-p should be between 0.1 and 1.0")
        except ValueError:
            print("Invalid top-p, keeping current value")
    rep_penalty = input(f"Repetition penalty [{settings['repetition_penalty']}]: ").strip()
    if rep_penalty:
        try:
            rep_val = float(rep_penalty)
            if 1.0 <= rep_val <= 2.0:
                settings['repetition_penalty'] = rep_val
            else:
                print("Repetition penalty should be between 1.0 and 2.0")
        except ValueError:
            print("Invalid repetition penalty, keeping current value")
    print("\nUpdated settings:")
    print_settings(settings)
    return settings
def parse_args():
    parser = argparse.ArgumentParser(description='Generate text using the trained model')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to start generation')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum length of generated text')
    return parser.parse_args()
def main():
    args = parse_args()    
    config = Config()
    model = GPTModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    tokenizer = HFTokenizer()
    tokenizer.load(config.tokenizer_path)
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length, device=config.device)    
    print("\nGenerated text:")
    print(generated_text)
if __name__ == '__main__':
    main()