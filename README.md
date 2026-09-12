# GPT From Scratch

A small character-level GPT trained on Shakespeare's works using PyTorch. The project is intentionally simple: it includes the data preparation step, a decoder-only Transformer, a training script, and a text-generation script.

## What it does

The model learns to predict the next character from the characters that came before it. After training, it can generate Shakespeare-like text from a short prompt.

This is an educational project rather than a production language model. The default configuration is designed for experimenting on a single machine and may take a while to train, especially without a CUDA-capable GPU.

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

The provided `requirements.txt` targets a CUDA-enabled PyTorch build. If you are using a CPU-only machine, install the appropriate PyTorch build from the [official PyTorch installation page](https://pytorch.org/get-started/locally/) before running the scripts.

## Run the project

Run these commands from the project root.

### 1. Prepare the data

```bash
python data/prepare.py
```

This reads `data/raw/shakespeare.txt`, builds a character vocabulary, and writes the encoded training and validation data to `data/processed/`.

### 2. Train the model

```bash
python train.py
```

Training progress is printed periodically. The latest model is saved to `checkpoints/model.pt`.

The main training settings live in `config.py`, including batch size, context length, model size, learning rate, and number of training iterations. The default model has about 10 million parameters.

### 3. Generate text

```bash
python generate.py
```

The script loads `checkpoints/model.pt`, starts from a newline prompt, prints generated text, and appends it to `outputs/samples.txt`.

## Project layout

```text
config.py                 Training and path configuration
train.py                  Model training and checkpointing
generate.py               Text generation from a checkpoint
data/prepare.py           Vocabulary creation and dataset preparation
model/gpt.py              GPT model and autoregressive generation
model/attention.py        Causal self-attention
model/transformer_block.py Transformer block components
data/raw/                 Source text
data/processed/           Generated dataset files and vocabulary
checkpoints/              Saved model checkpoints
outputs/                  Generated samples
```

## Notes

- The data and model paths in `config.py` are relative to the project root, so run the scripts from there.
- Training will overwrite `checkpoints/model.pt` as it progresses.
- Generation requires an existing checkpoint. Run training first if `checkpoints/model.pt` is missing.
- The model uses character-level tokenization, so it is compact and easy to inspect but less capable than a word- or subword-level language model.