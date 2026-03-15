"""Minimal training script for the tiny GPT language model."""

import os
import random

import torch
from torch.utils.data import DataLoader, random_split

from config import CONFIG
from dataset import CharLanguageModelingDataset
from model import GPTLanguageModel
from tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, loader: DataLoader, device: str, n_batches: int) -> float:
    """Estimate average loss on a few batches."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    """Train tiny GPT and save checkpoint/tokenizer."""
    set_seed(CONFIG.seed)

    if not os.path.exists(CONFIG.data_path):
        raise FileNotFoundError(
            f"Data file '{CONFIG.data_path}' not found. Create a plain text file first."
        )

    text = open(CONFIG.data_path, "r", encoding="utf-8").read()
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)

    dataset = CharLanguageModelingDataset(token_ids, block_size=CONFIG.block_size)
    train_size = int(len(dataset) * CONFIG.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG.batch_size)

    device = CONFIG.device if torch.cuda.is_available() else "cpu"

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=CONFIG.block_size,
        d_model=CONFIG.d_model,
        n_heads=CONFIG.n_heads,
        n_layers=CONFIG.n_layers,
        d_ff=CONFIG.d_ff,
        dropout=CONFIG.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)

    step = 0
    while step < CONFIG.max_steps:
        for x, y in train_loader:
            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B, T]

            _, loss, _ = model(x, targets=y)

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

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.vocab,
            "config": CONFIG.__dict__,
        },
        "checkpoint.pt",
    )
    print("Saved checkpoint to checkpoint.pt")


if __name__ == "__main__":
    main()
