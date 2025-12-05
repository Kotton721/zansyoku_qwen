#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ディレクトリ内の評価JSON（evalスクリプト出力）を読み、
(1) 各セクションの主要指標サマリ
(2) item+meal をまとめた「leaf結合」「subclass結合」のサマリ
を出力する。

抽出するキー（各セクション共通）:
  - num_items
  - exact_match_acc
  - mean_jaccard
  - micro_precision, micro_recall, micro_f1
  - macro_f1

出力ファイル:
  <元ファイル名>.summary.json
"""

from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict, Tuple, List

# 入力JSONのキーに揺れがある場合に対応（どちらでも可）
SECTION_KEY_CANDIDATES = {
    "item_leaf":      ["item_leaf", "item_leaf_level"],
    "meal_leaf":      ["meal_leaf", "meal_leaf_level"],
    "item_subclass":  ["item_subclass", "item_subclass_level"],
    "meal_subclass":  ["meal_subclass", "meal_subclass_level"],
}

METRICS = [
    "num_items",
    "exact_match_acc",
    "mean_jaccard",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_f1",
]

# 結合時に平均する対象（num_itemsは合計）
AVG_TARGETS = [
    "exact_match_acc",
    "mean_jaccard",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_f1",
]

def pick_section(obj: dict, logical_key: str) -> dict:
    """SECTION_KEY_CANDIDATES に基づいて、その論理キーに該当する最初の実キーを拾う"""
    for k in SECTION_KEY_CANDIDATES[logical_key]:
        if k in obj:
            return obj.get(k, {}) or {}
    return {}

def extract_metrics(sec: dict) -> dict:
    """必要なメトリクスだけ抽出"""
    if not sec:
        return {}
    out = {}
    for k in METRICS:
        if k in sec:
            out[k] = sec[k]
    return out

def combine_two_sections(a: dict, b: dict) -> dict:
    """
    2セクション（例: item_leaf と meal_leaf）を合算・平均して1ブロックにまとめる。
    - num_items は合計
    - それ以外は weighted（num_items重み）と macro（単純平均）の両方を出す
    出力キー例:
      num_items_total
      exact_match_acc_weighted / exact_match_acc_macro
      ...
    """
    out = {}
    n_a = float(a.get("num_items", 0) or 0)
    n_b = float(b.get("num_items", 0) or 0)
    n_tot = n_a + n_b

    out["num_items_total"] = int(n_tot)

    # macro平均（存在するものだけカウント）
    for m in AVG_TARGETS:
        vals = []
        if m in a and a[m] is not None:
            vals.append(float(a[m]))
        if m in b and b[m] is not None:
            vals.append(float(b[m]))
        if vals:
            out[f"{m}_macro"] = sum(vals) / len(vals)

    # 重み付き平均
    for m in AVG_TARGETS:
        v_a = float(a.get(m, 0) or 0)
        v_b = float(b.get(m, 0) or 0)
        if n_tot > 0:
            out[f"{m}_weighted"] = (v_a * n_a + v_b * n_b) / n_tot

    return out

def summarize_one(json_path: Path) -> Path:
    obj = json.loads(json_path.read_text(encoding="utf-8"))

    # 各セクション抜粋
    item_leaf      = extract_metrics(pick_section(obj, "item_leaf"))
    meal_leaf      = extract_metrics(pick_section(obj, "meal_leaf"))
    item_subclass  = extract_metrics(pick_section(obj, "item_subclass"))
    meal_subclass  = extract_metrics(pick_section(obj, "meal_subclass"))

    summary = {"source_file": json_path.name}

    settings = obj.get("settings", {})
    if settings:
        summary["settings"] = {
            "oov_policy": settings.get("oov_policy"),
            "subclass_source": settings.get("subclass_source"),
        }

    # そのままのブロック（従来どおり）
    summary["item_leaf"] = item_leaf
    summary["meal_leaf"] = meal_leaf
    summary["item_subclass"] = item_subclass
    summary["meal_subclass"] = meal_subclass

    # 追加: leaf 結合（item_leaf + meal_leaf）
    summary["leaf_combined"] = combine_two_sections(item_leaf, meal_leaf)

    # 追加: subclass 結合（item_subclass + meal_subclass）
    summary["subclass_combined"] = combine_two_sections(item_subclass, meal_subclass)

    out_path = json_path.with_suffix(".summary.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_perfile_summary.py /path/to/dir", file=sys.stderr)
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    if not in_dir.is_dir():
        print(f"[ERR] Not a directory: {in_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        print(f"[ERR] No .json files in {in_dir}", file=sys.stderr)
        sys.exit(1)

    for p in files:
        try:
            outp = summarize_one(p)
            print(f"[OK] {p.name} -> {outp.name}")
        except Exception as e:
            print(f"[WARN] skip {p.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
