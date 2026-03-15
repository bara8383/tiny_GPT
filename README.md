# tiny_GPT (PyTorch, minimal and readable)

Transformer内部理解のための、最小GPT風言語モデルです。

## 特徴

- `nn.Transformer` 不使用
- Hugging Face 不使用
- 既製 tokenizer 不使用（文字単位 tokenizer を自作）
- `batch_first=True` 相当の shape を統一利用: **`[batch, seq_len, d_model]`**
- causal mask の向きが明確
- attention weights を返却可能
- `generate.py` で `temperature` 指定可能
- 主要ハイパーパラメータは `config.py` で管理

## ファイル構成

- `config.py` : ハイパーパラメータ
- `tokenizer.py` : 文字単位 tokenizer
- `dataset.py` : next-token 用 Dataset
- `model.py` : attention / block / GPT本体
- `train.py` : 最小学習スクリプト
- `generate.py` : 生成スクリプト

## 1) データ準備

`data.txt` をプロジェクト直下に作成し、学習したいテキストを入れてください。

例:

```txt
hello tiny gpt.
this is a small demo.
```

## 2) 学習

```bash
python train.py
```

`checkpoint.pt` が保存されます。

## 3) 生成

```bash
python generate.py --prompt "hello" --max_new_tokens 80 --temperature 0.9
```

## テンソル shape の流れ

### 入力と埋め込み

1. `idx`: トークンID
   - shape: **`[B, T]`**
2. token embedding
   - `token_emb = Embedding(idx)`
   - shape: **`[B, T, C]`**
3. position embedding
   - `positions = arange(T)` → shape: `[T]`
   - `pos_emb = Embedding(positions)` → shape: `[T, C]`
   - broadcast で `[B, T, C]` 相当に加算
4. `x = token_emb + pos_emb`
   - shape: **`[B, T, C]`**

### Single-head causal self-attention

1. `q, k, v = Linear(x)`
   - shape: **`[B, T, head_dim]`**
2. スコア
   - `scores = q @ k^T / sqrt(head_dim)`
   - shape: **`[B, T, T]`**
3. causal mask（下三角）
   - `mask[i, j] = 1` if `j <= i`
   - 未来 (`j > i`) は `-inf` で無効化
4. `attn_weights = softmax(scores)`
   - shape: **`[B, T, T]`**
5. 出力
   - `out = attn_weights @ v`
   - shape: **`[B, T, head_dim]`**

### Multi-head

- 各head出力 `[B, T, head_dim]` を結合
- `concat -> [B, T, C]`
- `Linear` で `[B, T, C]`
- attention weights を返す場合: **`[B, n_heads, T, T]`**

### Transformer block

Pre-Norm + Residual:

1. `x = x + Attention(LN(x))`
2. `x = x + FFN(LN(x))`

各段で shape は **`[B, T, C]`** のまま。

### 出力と loss

1. `logits = lm_head(x)`
   - shape: **`[B, T, vocab_size]`**
2. target `y`
   - shape: **`[B, T]`**
3. Cross Entropy
   - `logits -> [B*T, vocab_size]`
   - `targets -> [B*T]`

## 学習タスク（next-token prediction）

Dataset は以下を返します。

- `x = data[i : i+block_size]`
- `y = data[i+1 : i+block_size+1]`

つまり、`y[t]` は `x[t]` の次トークンです。

## 補足

- あくまで理解用の最小実装です。
- 高性能化（学習率スケジューラ、weight decay調整、mixed precision、BPE等）は意図的に省いています。
