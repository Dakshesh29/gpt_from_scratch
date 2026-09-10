import json
from pathlib import Path
import numpy as np

#writing the paths of all the files we will be using in the project

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "raw" / "shakespeare.txt"
PROCESSED_DIR = DATA_DIR / "processed"

VOCAB_PATH = PROCESSED_DIR / "vocab.json"
TRAIN_PATH = PROCESSED_DIR / "train_data.npy"
VAL_PATH = PROCESSED_DIR / "val_data.npy"

# storing text data in a variable and then creating a set of all the unique characters in the text data and sorting them to create a vocabulary. The size of the vocabulary is then printed.

text = RAW_DATA_PATH.read_text(encoding="utf-8")

chars = sorted(list(set(text)))

vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")

# creating a mapping from char to index and index to char 

ctoi = {ch:i for i,ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}

# encoding the entire data into int 

def encode(text):
    return np.array([ctoi[c] for c in text], dtype=np.uint16)

# decoding the int data back into text
def decode(arr):
    return "".join([itoc[i] for i in arr])

txt_encoded = encode(text)

# splitting the data into training and validation sets

split = int(0.9 * len(txt_encoded))

train_data = txt_encoded[:split]
val_data = txt_encoded[split:]

# a dict to store vocab data

vocab_data = {
    "chars": chars
}


# creating the processed directory if it doesn't exist and then writing the vocab data to a json file and saving the training and validation data as numpy arrays.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# writing the vocab data to a json file and saving the training and validation data as numpy arrays.
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

#saving the training and validation data as numpy arrays
np.save(TRAIN_PATH, train_data)
np.save(VAL_PATH, val_data)

#checking if the encoding and decoding functions work correctly by asserting that decoding the encoded text returns the original text. Also, asserting that the total length of training and validation data equals the length of the encoded text.

assert decode(encode(text)) == text

assert len(train_data) + len(val_data) == len(txt_encoded)
