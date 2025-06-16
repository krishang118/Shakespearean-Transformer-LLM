import torch
import os
class Config:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        self.pin_memory = True if self.device.type == 'cuda' else False
        self.vocab_size = 12714  
        self.d_model = 768      
        self.n_heads = 12       
        self.n_layers = 8       
        self.d_ff = 3072       
        self.max_seq_len = 128  
        self.dropout = 0.2
        self.attention_dropout = 0.1
        self.embedding_dropout = 0.1
        self.layer_norm_eps = 1e-5
        self.batch_size = 32
        self.epochs = 15
        self.learning_rate = 1e-4     
        self.warmup_steps = 2000     
        self.weight_decay = 0.01
        self.gradient_clip = 1.0
        self.max_gen_length = 100
        self.temperature = 0.4
        self.top_k = 30
        self.top_p = 0.6
        self.repetition_penalty = 1.3
        self.beam_size = 4
        self.data_dir = "data"
        self.model_dir = "models"
        self.log_dir = "logs"
        self.tokenizer_path = os.path.join(self.model_dir, "final_model_tokenizer")
        self.model_path = os.path.join(self.model_dir, "final_model.pt")
        self.shakespeare_path = os.path.join(self.data_dir, "shakespeare.txt")
        self.label_smoothing = 0.1  
        self.early_stopping_patience = 10  
        self.train_split = 0.85     
        self.val_split = 0.15       
        self.min_text_length = 50   
        self.overlap_size = 20          
    def get_model_config(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout
        }    
    def get_training_config(self):
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'warmup_steps': self.warmup_steps,
            'grad_clip_norm': self.gradient_clip,
            'label_smoothing': self.label_smoothing
        }    
    def get_generation_config(self):
        return {
            'max_gen_length': self.max_gen_length,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'beam_size': self.beam_size
        }    
    def print_config(self):
        print("=== Configuration ===")
        print("Model Config:")
        for key, value in self.get_model_config().items():
            print(f"  {key}: {value}")
        print("\nTraining Config:")
        for key, value in self.get_training_config().items():
            print(f"  {key}: {value}")
        print("\nGeneration Config:")
        for key, value in self.get_generation_config().items():
            print(f"  {key}: {value}")
        print(f"\nDevice: {self.device}")
        print(f"Model save path: {self.model_path}")
        print(f"Tokenizer path: {self.tokenizer_path}")