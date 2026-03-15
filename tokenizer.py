"""Character-level tokenizer implementation for educational GPT."""

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """Simple character-level tokenizer.

    Attributes:
        vocab: Sorted list of unique characters.
        stoi: Character-to-index mapping.
        itos: Index-to-character mapping.
    """

    vocab: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build tokenizer vocabulary from raw text."""
        vocab = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(vocab=vocab, stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert string into list of token IDs."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to string."""
        return "".join(self.itos[i] for i in ids)
