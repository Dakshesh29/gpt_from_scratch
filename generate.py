import json
import torch

import config
from model import GPT


def load_vocab():
    with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    chars = vocab_data["chars"]
    ctoi = {ch: i for i, ch in enumerate(chars)}
    itoc = {i: ch for i, ch in enumerate(chars)}
    return ctoi, itoc


def encode(text, ctoi):
    return [ctoi[c] for c in text]


def decode(indices, itoc):
    return "".join(itoc[i] for i in indices)


def main():
    ctoi, itoc = load_vocab()

    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.device)

    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        block_size=checkpoint["block_size"],
        dropout=checkpoint["dropout"],
    ).to(config.device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prompt = "\n"
    context = torch.tensor([encode(prompt, ctoi)], dtype=torch.long, device=config.device)

    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    text = decode(generated, itoc)

    print(text)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SAMPLES_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n\n" + "=" * 40 + "\n\n")


if __name__ == "__main__":
    main()