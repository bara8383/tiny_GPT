# 実行環境セットアップガイド

このプロジェクトを動かすための最小手順です。  
`pip install pytorch` でうまくいかないケース（仮想環境を使ってください系のエラー）を避けるため、**最初に仮想環境を作成**します。

## 1. Python バージョン確認

```bash
python --version
```

Python 3.10 以上を推奨します。

## 2. 仮想環境の作成と有効化

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

有効化後は、プロンプト先頭に `(.venv)` が表示されます。

## 3. pip の更新

```bash
python -m pip install --upgrade pip
```

## 4. PyTorch のインストール

> 注意: パッケージ名は `pytorch` ではなく **`torch`** です。

CPU のみで動かす場合（まずはこれで OK）:

```bash
pip install torch
```

GPU (CUDA) を使う場合は、公式のインストール手順を参照して環境に合うコマンドを使ってください。

## 5. 動作確認

```bash
python -c "import torch; print(torch.__version__)"
python -m pytest -q
```

## 6. よくあるエラーと対処

### エラー例: 仮想環境を作成してください / externally-managed-environment

- 原因: システム Python に直接 `pip install` しようとしている。
- 対処: このドキュメントの「2. 仮想環境の作成と有効化」を実施してから再度インストール。

### エラー例: `No module named 'torch'`

- 原因: `torch` が未インストール、または別 Python に入っている。
- 対処:

```bash
which python
python -m pip show torch
```

`python` と `pip` が同じ仮想環境を指しているか確認してください。
