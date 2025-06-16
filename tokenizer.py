from tokenizers import ByteLevelBPETokenizer
import os
class HFTokenizer:
    def __init__(self, vocab_file=None, merges_file=None):
        if vocab_file and merges_file and os.path.exists(vocab_file) and os.path.exists(merges_file):
            self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        else:
            self.tokenizer = ByteLevelBPETokenizer()
    def train(self, text_path, vocab_size=30000):
        self.tokenizer.train(
            files=[text_path],
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<BOS>", "<EOS>"]
        )
    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenizer.encode(text)
        ids = tokens.ids
        if add_special_tokens:
            return [self.tokenizer.token_to_id("<BOS>")] + ids + [self.tokenizer.token_to_id("<EOS>")]
        return ids
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save_model(directory)
    def load(self, directory):
        vocab_file = os.path.join(directory, "vocab.json")
        merges_file = os.path.join(directory, "merges.txt")
        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("<PAD>")
    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<EOS>")