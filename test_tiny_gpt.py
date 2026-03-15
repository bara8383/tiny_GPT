import torch

from attention import MultiHeadCausalSelfAttention, SingleHeadCausalSelfAttention
from dataset import CharDataset
from model import GPTLanguageModel
from tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    text = "abc cab"
    tok = CharTokenizer()
    tok.fit(text)
    ids = tok.encode("abc")
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(tok.encode(text)) == text


def test_dataset_shift():
    tok = CharTokenizer()
    tok.fit("abcdef")
    ids = tok.encode("abcdef")
    ds = CharDataset(ids, seq_len=3)
    x, y = ds[1]
    assert x.shape == (3,)
    assert y.shape == (3,)
    assert torch.equal(x[1:], y[:-1])


def test_single_head_shapes_and_mask():
    attn = SingleHeadCausalSelfAttention(d_model=8, head_dim=4)
    x = torch.randn(2, 5, 8)
    out, w = attn(x, return_weights=True)
    assert out.shape == (2, 5, 4)
    assert w.shape == (2, 5, 5)
    # Future positions (upper triangle) should be ~0 after softmax
    upper = torch.triu(w[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_multi_head_shape():
    attn = MultiHeadCausalSelfAttention(d_model=12, n_heads=3)
    x = torch.randn(2, 4, 12)
    out, w = attn(x, return_weights=True)
    assert out.shape == (2, 4, 12)
    assert w.shape == (2, 3, 4, 4)


def test_model_logits_and_loss_shape():
    model = GPTLanguageModel(
        vocab_size=20,
        seq_len=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
    )
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    logits, loss, _ = model(x, y)
    assert logits.shape == (2, 8, 20)
    assert loss.ndim == 0
