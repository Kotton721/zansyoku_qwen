#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, csv, re, xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

# =============================
# 固定ラベル語彙（ユーザー提供の LABEL_SET）
# =============================
LABEL_SET = [
    # staple / grains / noodles
    "rice","bread","noodle","macaroni","udon","ramen_noodle","spaghetti",
    "potato","sweet_potato","taro",

    # meats / fish / processed
    "chicken","pork","beef","liver",
    "shishamo","salmon","mackerel","horse_mackerel","sardine","saury","mini_fish","kaeri_chirimen",
    "ham","sausage","bacon","kamaboko","chikuwa",

    # soy / egg / dairy
    "tofu","natto","aburaage","koya_tofu","soy_milk",
    "egg","quail_egg",
    "milk","cheese","yogurt",

    # processed dishes
    "hamburger_patty","shumai","gyoza","croquette","chicken_nugget","wonton",

    # leafy / fruit veg / roots
    "cabbage","hakusai","komatsuna","spinach","lettuce","mizuna",
    "pumpkin","eggplant","green_pepper","cucumber","okra","zucchini","tomato",
    "carrot","radish","burdock","lotus_root","onion",

    # mushrooms
    "shiitake","enoki","shimeji","maitake","eryngii","nameko",

    # seaweed
    "wakame","hijiki","kombu",

    # pulses & beans
    "edamame","green_pea","broad_bean","chickpea","lentil",

    # fruits & processed fruits
    "apple","pineapple","strawberry","orange","grape","banana","kiwi","melon","pear","peach",
    "fruit_compote","mixed_jam","apple_puree",

    # others
    "broccoli","soybeans","green_beans","red_beans","pickles",
    "green_onion","asparagus","butterbur","hanpen","bean_sprout",
]
VOCAB = set(LABEL_SET)

# =============================
# 軽い表記ゆれ正規化（必要なら拡張）
# =============================
NORM = {
    "green_peas": "green_pea",
    "spring_onion": "green_onion",
    "napa_cabbage": "hakusai",
    "chinese_cabbage": "hakusai",
    "sweetpotato": "sweet_potato",
    "wakamé": "wakame",
    "noodles": "noodle",
}

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = s.lower()
    s = s.replace("-", "_").replace(" ", "_")
    return NORM.get(s, s)

# =============================
# mealグループ化（__1 などの枝番を落とす）
# =============================
def meal_id_from_path(path: str) -> str:
    name = Path(path).name
    return re.sub(r"__\d+(?=\.[a-zA-Z0-9]+$)", "", name)

# =============================
# 予測JSONL読み込み
# JSONL: {"image": "/abs/or/relative/path.jpg", "remaining": ["rice", ...]}
# =============================
def read_pred_jsonl(p_path: Path) -> Dict[str, Set[str]]:
    """
    JSONLの1行=1JSONを基本としつつ、たまにある `}{` 連結も補正。
    画像キーは image_after > image > image_path > path > image_before の優先で取得。
    評価は常に「食後画像」を代表キーにしたいので image_after を最優先にする。
    """
    preds: Dict[str, Set[str]] = {}
    text = p_path.read_text(encoding="utf-8").strip()
    if not text:
        return preds

    # 連結されている場合 `}{` を改行で分割できるように補正
    text = re.sub(r'}\s*{', '}\n{', text)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        # 画像キーを柔軟に拾う（after を最優先）
        img = (
            obj.get("image_after")
            or obj.get("image")
            or obj.get("image_path")
            or obj.get("path")
            or obj.get("image_before")
        )
        if not img:
            # 画像パスが取れない行はスキップ
            continue

        labs = obj.get("remaining", []) or []
        labels = {_norm(x) for x in labs}
        preds[str(img)] = labels
    return preds

# =============================
# GT 読み込み（CSV/JSONL）
# CSV: image,labels  （labelsは;区切り）
# JSONL: {"image": "...", "labels": ["rice", ...]}
# =============================
def read_gt_csv_or_jsonl(gt_path: Path) -> Dict[str, Set[str]]:
    gt: Dict[str, Set[str]] = {}
    if gt_path.suffix.lower() == ".jsonl":
        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                labs = obj.get("labels", []) or []
                gt[obj["image"]] = {_norm(x) for x in labs}
    else:
        with gt_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row["image"]
                labs = row["labels"].split(";") if row["labels"].strip() else []
                gt[image] = {_norm(x) for x in labs}
    return gt

# =============================
# GT 読み込み（CVAT XML）
#  - 画像ごとのGT leaf集合
#  - <labels> から leaf→subclass マップ抽出
# =============================
def read_gt_cvat_xml_and_leaf2sub(xml_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # (A) leaf→subclass map
    leaf_to_subclass: Dict[str, str] = {}
    for lab in root.findall(".//labels/label"):
        name_el = lab.find("./name")
        if name_el is None or not name_el.text:
            continue
        leaf = _norm(name_el.text)
        subclass_val = None
        for attr in lab.findall("./attributes/attribute"):
            aname = attr.findtext("./name", default="")
            if _norm(aname) == "subclass":
                subclass_val = attr.findtext("./values", default="").strip()
                break
        if subclass_val:
            # valuesはカンマ区切りの可能性もあるので先頭を採用（必要なら集合化）
            subclass = _norm(subclass_val.split(",")[0])
            leaf_to_subclass[leaf] = subclass
        else:
            leaf_to_subclass[leaf] = "other"

    # (B) 画像ごとのGT leaf集合
    gt_leaf: Dict[str, Set[str]] = {}
    for img in root.iter("image"):
        name = img.attrib.get("name") or ""
        base = Path(name).name
        labels: Set[str] = set()
        # <tag label="...">
        for tag in img.findall("tag"):
            lab = tag.attrib.get("label")
            if lab:
                labels.add(_norm(lab))
        # 形状系（box, polygon, polyline, points, mask など）
        for shape in img:
            if isinstance(shape.tag, str) and shape.get("label"):
                labels.add(_norm(shape.get("label")))
        gt_leaf[base] = labels

    return gt_leaf, leaf_to_subclass

# =============================
# leaf→subclass 射影
# =============================
def to_subclass_set(leaf_set: Set[str], leaf_to_subclass: Dict[str, str]) -> Set[str]:
    out = set()
    for l in leaf_set:
        out.add(leaf_to_subclass.get(l, "other"))
    return out

# =============================
# 語彙外(OOV)処理（leaf用）
# =============================
def split_vocab(labels: Set[str]) -> Tuple[Set[str], Set[str]]:
    kept = {l for l in labels if l in VOCAB}
    oov  = {l for l in labels if l not in VOCAB}
    return kept, oov

def separate_vocab_for_all(d: Dict[str, Set[str]]):
    kept_d, oov_d = {}, {}
    all_oov = set()
    for k, labs in d.items():
        kept, oov = split_vocab(labs)
        kept_d[k] = kept
        oov_d[k] = oov
        all_oov |= oov
    return kept_d, oov_d, all_oov

# =============================
# 指標計算
# =============================
def prf_counts(y_true: Set[str], y_pred: Set[str]) -> Tuple[int,int,int]:
    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    return tp, fp, fn

def safe_div(a, b): 
    return a / b if b else 0.0

def exact_match(y_true: Set[str], y_pred: Set[str]) -> int:
    return int(y_true == y_pred)

def jaccard(y_true: Set[str], y_pred: Set[str]) -> float:
    u = len(y_true | y_pred)
    i = len(y_true & y_pred)
    return i / u if u else 1.0  # 両方空なら1

# predのキー（絶対パス）とGTのキー（相対名）をbasenameで合わせる
def align_keys(gt: Dict[str, Set[str]], pred: Dict[str, Set[str]]) -> Tuple[List[str], Dict[str, str]]:
    pred_by_base = {Path(k).name: k for k in pred.keys()}
    aligned_bases = []
    base2full = {}
    for gt_key in gt.keys():
        base = Path(gt_key).name
        if base in pred_by_base:
            aligned_bases.append(base)
            base2full[base] = pred_by_base[base]
    return aligned_bases, base2full

def evaluate_itemwise(
    gt: Dict[str, Set[str]],
    pred: Dict[str, Set[str]],
    gt_oov_map: Dict[str, Set[str]],
    pred_oov_map: Dict[str, Set[str]],
    oov_policy: str = "ignore"
):
    bases, base2full = align_keys(gt, pred)

    ex_total = 0
    jac_total = 0.0
    TP = FP = FN = 0

    per_class = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))

    # OOVペナルティのための累積
    oov_FP = oov_FN = 0

    for base in bases:
        y_t = gt[base]
        y_p = pred[base2full[base]]

        ex_total += exact_match(y_t, y_p)
        jac_total += jaccard(y_t, y_p)

        tp, fp, fn = prf_counts(y_t, y_p)
        TP += tp; FP += fp; FN += fn

        for c in (y_t & y_p):
            per_class[c]["tp"] += 1
        for c in (y_p - y_t):
            per_class[c]["fp"] += 1
        for c in (y_t - y_p):
            per_class[c]["fn"] += 1

        if oov_policy == "penalize":
            # 予測OOVはFPとして、GT OOVはFNとして加点（ラベル名は未知なので per_class には足さない）
            oov_FP += len(pred_oov_map.get(base2full[base], set()))
            oov_FN += len(gt_oov_map.get(base, set()))

    # OOVペナルティ反映（マイクロ指標にのみ影響）
    TP_final = TP
    FP_final = FP + (oov_FP if oov_policy == "penalize" else 0)
    FN_final = FN + (oov_FN if oov_policy == "penalize" else 0)

    micro_p = safe_div(TP_final, TP_final + FP_final)
    micro_r = safe_div(TP_final, TP_final + FN_final)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r) if (micro_p + micro_r) else 0.0

    macro_f1s = []
    for c, cnt in per_class.items():
        p = safe_div(cnt["tp"], cnt["tp"] + cnt["fp"])
        r = safe_div(cnt["tp"], cnt["tp"] + cnt["fn"])
        f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
        macro_f1s.append(f1)
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    n = len(bases)
    report = {
        "num_items": n,
        "exact_match_acc": safe_div(ex_total, n),
        "mean_jaccard": safe_div(jac_total, n),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class": {
            c: {
                "tp": int(cnt["tp"]),
                "fp": int(cnt["fp"]),
                "fn": int(cnt["fn"]),
                "precision": safe_div(cnt["tp"], cnt["tp"] + cnt["fp"]),
                "recall": safe_div(cnt["tp"], cnt["tp"] + cnt["fn"]),
                "f1": (lambda p, r: safe_div(2*p*r, p+r) if (p+r) else 0.0)(
                    safe_div(cnt["tp"], cnt["tp"] + cnt["fp"]),
                    safe_div(cnt["tp"], cnt["tp"] + cnt["fn"]),
                ),
            }
            for c, cnt in per_class.items()
        },
        "oov_penalty": {
            "applied": (oov_policy == "penalize"),
            "pred_oov_as_FP": int(oov_FP) if oov_policy == "penalize" else 0,
            "gt_oov_as_FN": int(oov_FN) if oov_policy == "penalize" else 0,
        }
    }
    return report

def group_by_meal(d: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    g = defaultdict(set)
    for path_or_base, labs in d.items():
        base = Path(path_or_base).name
        g[meal_id_from_path(base)].update(labs)
    return dict(g)

def group_oov_by_meal(oov_map: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    g = defaultdict(set)
    for path_or_base, labs in oov_map.items():
        base = Path(path_or_base).name
        g[meal_id_from_path(base)].update(labs)
    return dict(g)

# =============================
# メイン
# =============================
def main():
    ap = argparse.ArgumentParser(description="Evaluate VLM 'remaining food' with leaf & subclass metrics.")
    ap.add_argument("--pred", required=True, help="predictions JSONL (from run_qwen_vl_remaining.py)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gt", help="ground truth CSV or JSONL path (image,labels)")
    g.add_argument("--gt_cvat_xml", help="ground truth CVAT XML path (image/tag etc.)")
    ap.add_argument("--subclass_xml", help="XML that defines <labels> with subclass attribute (optional if --gt_cvat_xml given)")
    ap.add_argument("--report_prefix", default="eval_report", help="output JSON prefix")
    ap.add_argument("--oov_policy", choices=["ignore","penalize"], default="ignore", help="OOV handling for leaf-level")
    ap.add_argument("--out", default=None, help="出力ファイル名を直接指定（例: result.json, - はstdout）")

    args = ap.parse_args()

    # ---- 予測読み込み ----
    pred_raw = read_pred_jsonl(Path(args.pred))   # keys are full paths

    # ---- GT & leaf->subclass ----
    leaf2sub: Dict[str, str] = {}

    if args.gt_cvat_xml:
        gt_leaf, leaf2sub = read_gt_cvat_xml_and_leaf2sub(Path(args.gt_cvat_xml))
        gt_leaf = {Path(k).name: v for k, v in gt_leaf.items()}  # basenameに揃える
    else:
        # GTはCSV/JSONL
        gt_leaf = read_gt_csv_or_jsonl(Path(args.gt))
        gt_leaf = {Path(k).name: v for k, v in gt_leaf.items()}
        # subclass 定義は別XMLから取得（必須推奨）
        if args.subclass_xml:
            _, leaf2sub = read_gt_cvat_xml_and_leaf2sub(Path(args.subclass_xml))
        else:
            # 与えられない場合は unknown を other とする
            leaf2sub = defaultdict(lambda: "other")

    # ---- 語彙内/外に分離（leaf用）----
    pred_kept, pred_oov_map, pred_all_oov = separate_vocab_for_all(pred_raw)
    gt_kept, gt_oov_map, gt_all_oov = separate_vocab_for_all(gt_leaf)

    if pred_all_oov:
        print(f"[WARN] OOV in predictions ({len(pred_all_oov)} kinds, ignored unless penalize): {sorted(pred_all_oov)}")
    if gt_all_oov:
        print(f"[WARN] OOV in GT ({len(gt_all_oov)} kinds, ignored unless penalize): {sorted(gt_all_oov)}")

    # ---- 画像（皿）単位：leaf評価 ----
    item_leaf_report = evaluate_itemwise(
        gt=gt_kept,
        pred=pred_kept,
        gt_oov_map=gt_oov_map,
        pred_oov_map=pred_oov_map,
        oov_policy=args.oov_policy
    )

    # ---- 食事（グループ）単位（和集合）：leaf評価 ----
    pred_meal = group_by_meal(pred_kept)
    gt_meal   = group_by_meal(gt_kept)
    pred_meal_oov = group_oov_by_meal(pred_oov_map)
    gt_meal_oov   = group_oov_by_meal(gt_oov_map)

    meal_leaf_report = evaluate_itemwise(
        gt=gt_meal,
        pred=pred_meal,
        gt_oov_map=gt_meal_oov,
        pred_oov_map=pred_meal_oov,
        oov_policy=args.oov_policy
    )

    # ---- subclass 射影（item） ----
    def proj_dict_to_sub(d: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        return {k: to_subclass_set(v, leaf2sub) for k, v in d.items()}

    item_sub_report = evaluate_itemwise(
        gt=proj_dict_to_sub(gt_kept),
        pred=proj_dict_to_sub(pred_kept),
        gt_oov_map={},  # 射影後OOVなし扱い
        pred_oov_map={},
        oov_policy="ignore"
    )

    # ---- subclass 射影（meal） ----
    meal_sub_report = evaluate_itemwise(
        gt=proj_dict_to_sub(gt_meal),
        pred=proj_dict_to_sub(pred_meal),
        gt_oov_map={},
        pred_oov_map={},
        oov_policy="ignore"
    )

    out = {
    "settings": {
        "oov_policy": args.oov_policy,
        "subclass_source": args.gt_cvat_xml or args.subclass_xml or "(none)",
    },
    "item_leaf_level": item_leaf_report,
    "meal_leaf_level": meal_leaf_report,
    "item_subclass_level": item_sub_report,
    "meal_subclass_level": meal_sub_report,
    }

    # ---- 出力先を決定 ----
    if args.out and args.out.strip() == "-":
        # 標準出力
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = Path(f"{args.report_prefix}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
