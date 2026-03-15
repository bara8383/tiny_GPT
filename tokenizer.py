"""Character-level tokenizer for tiny GPT."""


class CharTokenizer:
    """A tiny character-level tokenizer.

    Stores:
    - stoi: string(char) -> int(id)
    - itos: int(id) -> string(char)
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def fit(self, text: str) -> None:
        """Build vocabulary from raw text."""
        vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        if not self.stoi:
            raise ValueError("Tokenizer is not fitted. Call fit(text) first.")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""
        if not self.itos:
            raise ValueError("Tokenizer is not fitted. Call fit(text) first.")
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.stoi)

    def state_dict(self) -> dict:
        """Return serializable tokenizer state."""
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        """Restore tokenizer from saved state."""
        tok = cls()
        tok.stoi = {str(k): int(v) for k, v in state["stoi"].items()}
        tok.itos = {int(k): str(v) for k, v in state["itos"].items()}
        return tok
