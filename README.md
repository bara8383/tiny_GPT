# tiny_GPT

## 0. 実行環境の準備

環境構築手順は `SETUP.md` にまとめています。
特に `pip install pytorch` ではなく `pip install torch` を使う点と、先に仮想環境を作る点を確認してください。

## 1. このプロジェクトの目的

このリポジトリは、**最小構成の GPT 風言語モデル**を PyTorch で実装し、
次トークン予測（next-token prediction）と自己回帰生成の流れを学ぶことを目的としています。

- 文字単位 tokenizer
- causal self-attention
- multi-head attention
- Transformer block
- 学習スクリプト
- 生成スクリプト

を段階的に確認できます。

## 2. ディレクトリ構成

- `config.py` : ハイパーパラメータ設定
- `tokenizer.py` : 文字単位 tokenizer (`fit/encode/decode`)
- `dataset.py` : `x, y` の next-token 用データセット
- `attention.py` : single-head / multi-head causal self-attention
- `block.py` : FeedForward と TransformerBlock (Pre-LN)
- `model.py` : GPT 本体
- `train.py` : 学習
- `generate.py` : 自己回帰生成
- `test_tiny_gpt.py` : 最小テスト

## 3. 学習方法

1. 学習テキストを `input.txt` に置く
2. 学習実行

```bash
python train.py
```

出力:
- `model.pt`
- `tokenizer.json`
- `model_config.json`

## 4. 生成方法

### sampling

```bash
python generate.py --prompt "hello" --max_new_tokens 80 --temperature 0.9 --do_sample
```

### greedy

```bash
python generate.py --prompt "hello" --max_new_tokens 80 --temperature 1.0 --greedy
```

## 5. 各テンソル shape の流れ

- input ids: `[batch, seq_len]`
- token embedding: `[batch, seq_len, d_model]`
- positional embedding 加算後: `[batch, seq_len, d_model]`
- attention scores: `[batch, n_heads, seq_len, seq_len]`
- block output: `[batch, seq_len, d_model]`
- logits: `[batch, seq_len, vocab_size]`

### 補足説明

- **causal mask とは何か**  
  各時刻 `t` が `t` より未来の token を見ないようにするマスクです。実装では下三角を残し、右上三角（未来側）を潰します。

- **なぜ future token を見てはいけないのか**  
  学習時と生成時で同じ条件（過去と現在のみ利用）を保つためです。未来情報を見てしまうと不正な学習になります。

- **next-token prediction とは何か**  
  `x` の各位置から「次の 1 文字」を当てるタスクです。  
  例: `x = text[i : i+seq_len]`, `y = text[i+1 : i+seq_len+1]`

## 推奨初期値

```python
vocab_size = None
seq_len = 64
batch_size = 32
d_model = 128
n_heads = 4
n_layers = 2
d_ff = 256
dropout = 0.1
learning_rate = 3e-4
max_steps = 2000
eval_interval = 100
device = "cuda"
```

CPU 向けに小さくするなら:

```python
seq_len = 32
batch_size = 16
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 128
```
