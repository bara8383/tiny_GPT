"""Autoregressive text generation script for tiny GPT."""

import argparse
import json

import torch

from model import GPTLanguageModel
from tokenizer import CharTokenizer


def load_artifacts(model_path: str, tokenizer_path: str, config_path: str, device: str):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_state = json.load(f)
    tokenizer = CharTokenizer.from_state_dict(tok_state)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer, cfg


@torch.no_grad()
def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
) -> str:
    ids = tokenizer.encode(prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len :]
        logits, _, _ = model(idx_cond)

        # We only use the last time-step logits because that predicts the next token.
        next_logits = logits[:, -1, :] / temperature

        if do_sample:
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

        # Append one token and repeat (autoregressive generation).
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--config", default="model_config.json")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_artifacts(args.model, args.tokenizer, args.config, device)

    do_sample = args.do_sample
    if args.greedy:
        do_sample = False

    out = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=do_sample,
        device=device,
    )
    print(out)


if __name__ == "__main__":
    main()
