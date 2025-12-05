#!/bin/bash
# ============================================
# Qwen-VL 残食推定バッチ推論スクリプト
# モデル: Qwen3B / Qwen7B / Qwen32B
# 出力: /home/user/qwen/convert/results/
# ============================================

set -e

PAIRS_CSV="/home/user/qwen/qwen_ab_pred/pairs_new.csv"
XML_PATH="/home/user/qwen/qwen_after_only/datasets/indi_image/images/after_img_all_xml/annotations.xml"
OUT_DIR="/home/user/qwen/qwen_ab_pred/ab_results"

# 共通設定
COMMON_ARGS="--pairs_csv $PAIRS_CSV \
  --before-col before_new \
  --after-col after_new \
  --xml $XML_PATH \
  --device_map auto \
  --dtype auto \
  --max_new_tokens 128 \
  --temperature 0.0"

# ===== 実行関数 =====
run_model () {
  MODEL_SUFFIX="$1"
  MODEL_PATH="$2"
  OUT_NAME="Qwen${MODEL_SUFFIX}"

  echo "===================================="
  echo "[RUNNING] Model: $OUT_NAME"
  echo "===================================="

  python pred_qwen_ab_ingred.py \
    $COMMON_ARGS \
    --model "$MODEL_PATH" \
    --out-jsonl "$OUT_DIR/${OUT_NAME}.jsonl" \
    --out-csv "$OUT_DIR/${OUT_NAME}.csv"

  echo "[DONE] $OUT_NAME の推論結果を保存しました"
  echo
}

# ===== 実行対象モデル =====
run_model "3B"  "Qwen/Qwen2.5-VL-3B-Instruct"
run_model "7B"  "Qwen/Qwen2.5-VL-7B-Instruct"
run_model "32B" "Qwen/Qwen2.5-VL-32B-Instruct"

echo "===================================="
echo "すべての推論が完了しました。出力先: $OUT_DIR"
echo "===================================="
