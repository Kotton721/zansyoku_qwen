# qwen-urushi

Qwen2.5-VL を使って「食前・食後の画像から何が残っているか」を推定する実験リポジトリです。用途に応じて以下の 2 系統を同梱しています。
- `qwen/qwen_ab_pred`: 食前 + 食後のペア画像を比較して残食を推定。
- `qwen/qwen_after_only`: 食後画像だけから残っている食材を推定。

## 環境とセットアップ
- Python 3.10+ / GPU(CUDA) 前提。`requirements.txt` は PyTorch nightly (CUDA 12.8) を向いています。
- ローカル環境なら `python3 -m venv .venv && source .venv/bin/activate` の上で `pip install -r requirements.txt` を実行。
- まとまった依存を隔離したい場合は同梱の `Dockerfile` / `docker-compose.yml` も利用できます（中で上記の要件を入れる想定）。

## データ前提
- ラベル定義は CVAT 形式の XML から取得します（`<labels><label><name>` と `<attribute name="subclass">` を利用）。
- 画像は `before` / `after` ディレクトリに振り分け、ファイル名で同一食事を紐づける形を想定しています（具体例は各サブ README を参照）。

## 使い方の流れ
- どちらのワークフローを使うか決めて、詳細は各ディレクトリの README を参照してください。
  - ペア比較推論: `qwen/qwen_ab_pred/README.md`
  - 食後のみ推論: `qwen/qwen_after_only/README.md`
- 共通で、推論スクリプトが JSONL/CSV の結果を吐き、`eval.py` で leaf/subclass の精度指標を計算できます。

## ディレクトリ概要
- `qwen/qwen_ab_pred`: ペア画像推論、CSV ペア作成・整備スクリプト、ペア用の評価コード。
- `qwen/qwen_after_only`: 食後単体推論、データ整形スクリプト、サブクラス別集計や評価コード。
- `requirements.txt`: 推論に必要な Python パッケージ（Torch, transformers など）。
- `Dockerfile` / `docker-compose.yml`: GPU 環境用のコンテナ雛形。
