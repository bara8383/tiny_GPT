"""Configuration values for tiny GPT."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # data
    data_path: str = "input.txt"
    train_split: float = 0.9
    vocab_size: int | None = None
    seq_len: int = 64

    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 2000
    eval_interval: int = 100
    eval_batches: int = 20

    # generation
    max_new_tokens: int = 120
    temperature: float = 1.0
    do_sample: bool = True

    # runtime
    seed: int = 42
    device: str = "cuda"


CONFIG = GPTConfig()
