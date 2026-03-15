"""Training script for tiny GPT."""

import json
import os
import random

import torch
from torch.utils.data import DataLoader, random_split

from config import CONFIG
from dataset import CharDataset
from model import GPTLanguageModel
from tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, loader: DataLoader, device: str, n_batches: int) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    set_seed(CONFIG.seed)

    if not os.path.exists(CONFIG.data_path):
        raise FileNotFoundError(f"{CONFIG.data_path} not found")

    text = open(CONFIG.data_path, "r", encoding="utf-8").read()

    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    token_ids = tokenizer.encode(text)

    dataset = CharDataset(token_ids, seq_len=CONFIG.seq_len)
    train_size = int(len(dataset) * CONFIG.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG.batch_size)

    device = CONFIG.device if (CONFIG.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        seq_len=CONFIG.seq_len,
        d_model=CONFIG.d_model,
        n_heads=CONFIG.n_heads,
        n_layers=CONFIG.n_layers,
        d_ff=CONFIG.d_ff,
        dropout=CONFIG.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)

    model.train()
    step = 0
    while step < CONFIG.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits, loss, _ = model(x, targets=y)
            _ = logits  # keep explicit for readability

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step % CONFIG.eval_interval == 0:
                train_loss = estimate_loss(model, train_loader, device, CONFIG.eval_batches)
                val_loss = estimate_loss(model, val_loader, device, CONFIG.eval_batches)
                print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            if step >= CONFIG.max_steps:
                break

    torch.save(model.state_dict(), "model.pt")
    with open("tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.state_dict(), f, ensure_ascii=False, indent=2)
    with open("model_config.json", "w", encoding="utf-8") as f:
        cfg = CONFIG.__dict__.copy()
        cfg["vocab_size"] = tokenizer.vocab_size
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("Saved model.pt, tokenizer.json, model_config.json")


if __name__ == "__main__":
    main()
