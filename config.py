"""Central configuration for the tiny GPT example."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyper-parameters for model, data and training.

    All tensor shapes in this project use batch-first format:
    [batch, seq_len, d_model].
    """

    # data
    data_path: str = "data.txt"
    block_size: int = 64  # context length (seq_len)
    train_split: float = 0.9

    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 20

    # generation
    max_new_tokens: int = 120
    temperature: float = 1.0

    # runtime
    seed: int = 42
    device: str = "cpu"


CONFIG = GPTConfig()
