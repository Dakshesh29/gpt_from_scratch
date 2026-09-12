import json
import torch
import numpy as np

import config
from model import GPT


def load_data():
    train_data = np.load(config.TRAIN_PATH)
    val_data = np.load(config.VAL_PATH)
    return (
        torch.tensor(train_data, dtype=torch.long),
        torch.tensor(val_data, dtype=torch.long),
    )


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]) 
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(data, config.block_size, config.batch_size, config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data["chars"])

    train_data, val_data = load_data()

    model = GPT(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
    ).to(config.device)

    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_layer": config.n_layer,
                "block_size": config.block_size,
                "dropout": config.dropout,
            }, config.CHECKPOINT_PATH)

        xb, yb = get_batch(train_data, config.block_size, config.batch_size, config.device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete. Final checkpoint saved to", config.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()