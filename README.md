# Shakespearean Language Model

A transformer-based language model trained on Shakespeare's works, capable of generating text in the style of the Bard. This project implements a custom transformer architecture and training pipeline from scratch, focusing on generating coherent and stylistically accurate Shakespearean text.

## 🎭 Features

- Custom transformer architecture implementation
- BPE (Byte Pair Encoding) tokenizer trained on Shakespeare's works
- Coherence-focused training with custom loss functions
- Text generation with temperature and top-p sampling
- Character-aware generation preserving dialogue structure
- Early stopping and model checkpointing
- MPS (Metal Performance Shaders) support for Apple Silicon

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+
- torchtext 0.15+
- tokenizers 0.15+
- tqdm 4.64+

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
.
├── config.py           # Configuration settings
├── generate.py         # Text generation script
├── train.py           # Model training script
├── tokenizer.py       # Custom tokenizer implementation
├── data/              # Training data and tokenizer files
│   └── shakespeare.txt
├── models/            # Trained models and tokenizers
│   ├── final_model.pt
│   └── final_model_tokenizer/
└── requirements.txt   # Project dependencies
```

## 📥 Pre-trained Model

The pre-trained model is available for download from Google Drive:

[Download Pre-trained Model (final_model.pt)](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)

After downloading:
1. Create a `models` directory in the project root
2. Place the downloaded `final_model.pt` file in the `models` directory
3. The model will be automatically loaded when running `generate.py`

Note: The model file is approximately 254MB. Make sure you have sufficient storage space.

## 🚀 Usage

### Training the Model

To train the model from scratch:

```bash
python train.py
```

The training script will:
1. Train a custom BPE tokenizer on Shakespeare's works
2. Initialize the transformer model
3. Train the model with early stopping
4. Save the best model and tokenizer

### Generating Text

To generate Shakespearean text:

```bash
python generate.py --prompt "HAMLET: To be, or not to be, that is the question"
```

Example prompts that work well:
1. `"HAMLET: To be, or not to be, that is the question"`
2. `"JULIET: O Romeo, Romeo! wherefore art thou Romeo?"`
3. `"MACBETH: Is this a dagger which I see before me?"`
4. `"PORTIA: The quality of mercy is not strain'd"`
5. `"IAGO: O, beware, my lord, of jealousy"`

## 🎯 Model Architecture

The model uses a transformer architecture with:
- 6 transformer layers
- 8 attention heads
- 512 embedding dimensions
- 2048 feed-forward dimensions
- Learned positional embeddings
- Layer normalization
- Dropout for regularization

## 🎨 Generation Parameters

The model uses the following parameters for text generation:
- Temperature: 0.11 (controls randomness)
- Top-p: 0.89 (nucleus sampling)
- Max length: 100 tokens
- No repetition penalty

## 📊 Training Details

- Batch size: 32
- Learning rate: 3e-4 with warmup
- Early stopping patience: 5 epochs
- Validation split: 15%
- Training epochs: 15 (with early stopping)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Shakespeare's works for the training data
- PyTorch team for the deep learning framework
- Hugging Face for the transformers library