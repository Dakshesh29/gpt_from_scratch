import torch
from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

VOCAB_PATH = PROCESSED_DIR / "vocab.json"
TRAIN_PATH = PROCESSED_DIR / "train_data.npy"
VAL_PATH = PROCESSED_DIR / "val_data.npy"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pt"

OUTPUT_DIR = Path("outputs")
SAMPLES_PATH = OUTPUT_DIR / "samples.txt"

block_size = 256      
n_embd = 384           
n_head = 6            
n_layer = 6          
dropout = 0.2


batch_size = 64
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200      

device = "cuda" if torch.cuda.is_available() else "cpu"