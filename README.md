# Shakespearean Transformer-based LLM

A from-scratch transformer-based large language model that is trained on, and generates Shakespearean text. This project implements a custom transformer architecture and training pipeline, focusing on generating coherent and stylistically accurate text in the style of the Bard.

## Features

- Custom transformer architecture implementation
- Hugging Face's ByteLevelBPE tokenizer trained on Shakespeare's works
- Coherence-focused training with custom loss functions
- Text generation with optimized parameters (temperature: 0.11, top-p: 0.89)
- Character-aware generation preserving dialogue structure
- Early stopping and model checkpointing
- GPU acceleration support (CUDA for NVIDIA GPUs, MPS for Apple Silicon)

## Requirements

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

## Project Structure

```
├── config.py             # Configuration settings
├── generate.py           # Text generation script
├── train.py              # Model training script
├── tokenizer.py          # Tokenizer implementation
├── data/                 # Training data and tokenizer files
│   ├── shakespeare.txt
│   └── tokenizer/        
├── models/               # Trained models and tokenizers
│   ├── final_model.pt    # Train separately or download from provided link
│   └── final_model_tokenizer/
└── requirements.txt      # Project dependencies
```

## Pre-trained Model

The pre-trained model is available for download from Google Drive:

[Download Pre-trained Model (final_model.pt)](https://drive.google.com/file/d/1Uta3DRXaVfUdRgS51vrhIU37pbtitRay/view?usp=sharing)

After downloading:
1. Place the downloaded `final_model.pt` file in the `models` directory, present in the main project directory.
2. The model will automatically be loaded when running `generate.py`.

## Usage

### Training the Model

After you have cloned this repository on your local machine, you would have to train the model from scratch yourself, or download and load the pre-trained model.
To train the model from scratch:

```bash
python train.py
```

The training script will:
1. Train the ByteLevelBPE tokenizer on Shakespeare's works
2. Initialize the transformer model
3. Train the model (till 15 epochs, with learning rate 1e-4, and batch size 32) with early stopping
4. Save the model and tokenizer

### Generating Text

An example of generating Shakespearean text:

```bash
python generate.py --prompt "HAMLET: To be, or not to be, that is the question"
```

## Model Architecture

The model uses a transformer architecture with:
- 8 transformer layers
- 12 attention heads
- 768 embedding dimensions
- 3072 feed-forward dimensions
- Learned positional embeddings
- Layer normalization
- Dropout for regularization

## Generation Parameters

The model uses the following default parameters for text generation:
- Temperature: 0.11 (controls randomness)
- Top-p: 0.89 (nucleus sampling)
- Max length: 100 tokens

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License.  
