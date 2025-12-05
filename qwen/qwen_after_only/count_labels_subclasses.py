#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAT XML から
- 定義されているラベル（クラス）数
- 定義されているサブクラス数（subclass属性のユニーク数）
- ついでに画像内で実際に使われたラベル／サブクラス数
を集計します。

使い方:
  python count_labels_subclasses.py /path/to/annotations.xml
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    return s

def main(xml_path: str):
    path = Path(xml_path)
    if not path.exists():
        print(f"[ERR] file not found: {path}")
        sys.exit(1)

    tree = ET.parse(path)
    root = tree.getroot()

    # ========== (1) <labels> 定義の収集 ==========
    defined_labels = []                 # 例: ["rice","chicken",...]
    label_to_subclass = {}              # 例: {"rice":"grain", ...}
    subclass_values_all = []            # 生のsubclass値（カンマで複数ある場合は分割）

    for lab in root.findall(".//labels/label"):
        name_el = lab.find("./name")
        if name_el is None or not name_el.text:
            continue
        label = _norm(name_el.text)
        defined_labels.append(label)

        # subclass属性を探す
        subclass_val = None
        for attr in lab.findall("./attributes/attribute"):
            aname = _norm(attr.findtext("./name", default=""))
            if aname == "subclass":
                # values には複数値が入ることがあるので全取得
                raw = (attr.findtext("./values", default="") or "").strip()
                if raw:
                    # カンマ・改行・セミコロンなどを区切りとして扱う
                    parts = re.split(r"[,;\n]+", raw)
                    parts = [_norm(p) for p in parts if p.strip()]
                    subclass_values_all.extend(parts)
                    # 代表値（先頭）をマップに入れる（あなたの既存スクリプトと同じ方針）
                    subclass_val = parts[0]
                break
        if subclass_val:
            label_to_subclass[label] = subclass_val
        else:
            # 未定義は other 扱い（必要なら変更）
            label_to_subclass[label] = "other"

    defined_label_set = set(defined_labels)
    defined_subclass_set = set(subclass_values_all) if subclass_values_all else set()

    # ========== (2) 画像側で実際に使われたラベル（tag/shape） ==========
    used_label_set = set()
    for img in root.iter("image"):
        # <tag label="...">
        for tag in img.findall("tag"):
            lab = tag.attrib.get("label")
            if lab:
                used_label_set.add(_norm(lab))
        # 形状（box, polygon, polyline, points, mask など）
        for shape in img:
            if isinstance(shape.tag, str) and shape.get("label"):
                used_label_set.add(_norm(shape.get("label")))

    # 画像で使われたサブクラス集合（label_to_subclass で射影）
    used_subclass_set = set(label_to_subclass.get(l, "other") for l in used_label_set)

    # ========== (3) 出力 ==========
    print("=== From <labels> definition ===")
    print(f"Defined classes (labels): {len(defined_label_set)}")
    print(f"Defined subclasses (unique in 'subclass' values): {len(defined_subclass_set)}")
    if defined_subclass_set:
        # 表示が多すぎると邪魔なので上位いくつか
        sample = sorted(defined_subclass_set)[:20]
        more = "" if len(defined_subclass_set) <= 20 else f" ... (+{len(defined_subclass_set)-20})"
        print(f"Subclasses sample: {sample}{more}")

    # サブクラスごとの属するラベル数
    sub2labels = defaultdict(list)
    for lab, sub in label_to_subclass.items():
        sub2labels[sub].append(lab)

    print("\nSubclass -> #labels  (top 20):")
    for sub, labs in sorted(sub2labels.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        print(f"  {sub:20s} : {len(labs)}")

    print("\n=== From <image> usage (annotations) ===")
    print(f"Used classes in images: {len(used_label_set)}")
    print(f"Used subclasses in images: {len(used_subclass_set)}")

    # 使われたが定義に無いラベルの検出（CVATで時々ある）
    undefined_used = sorted(used_label_set - defined_label_set)
    if undefined_used:
        print(f"\n[WARN] Labels used in images but not defined in <labels>: {len(undefined_used)}")
        print(f"  {undefined_used[:20]}{' ...' if len(undefined_used)>20 else ''}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_labels_subclasses.py /path/to/annotations.xml")
        sys.exit(1)
    main(sys.argv[1])
