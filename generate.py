"""Text generation script for the tiny GPT language model."""

import argparse

import torch

from config import GPTConfig
from model import GPTLanguageModel
from tokenizer import CharTokenizer


def load_model(checkpoint_path: str, device: str) -> tuple[GPTLanguageModel, CharTokenizer, GPTConfig]:
    """Load model/tokenizer/config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = GPTConfig(**checkpoint["config"])

    vocab = checkpoint["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    tokenizer = CharTokenizer(vocab=vocab, stoi=stoi, itos=itos)

    model = GPTLanguageModel(
        vocab_size=len(vocab),
        block_size=cfg.block_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, cfg


def main() -> None:
    """Generate continuation text from a prompt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_model(args.checkpoint, device)

    max_new_tokens = args.max_new_tokens or cfg.max_new_tokens
    temperature = args.temperature if args.temperature is not None else cfg.temperature

    input_ids = tokenizer.encode(args.prompt)
    if not input_ids:
        input_ids = [0]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, T]
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

    generated_text = tokenizer.decode(out[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
