# pred_qwen_ab_ingred.py
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # Qwen 付属

# ====== ラベル集合はXMLからロード ======
LABEL_SET: list[str] = []

# ====== 軽量シノニム（必要に応じて追加） ======
SYNONYMS = {
    # スペース/ハイフン差異
    "ramen noodle": "ramen_noodle",
    "horse mackerel": "horse_mackerel",
    "lotus root": "lotus_root",
    "green onion": "green_onion",
    "green pepper": "green_pepper",
    "apple puree": "apple_puree",
    "fruit compote": "fruit_compote",
    "mixed jam": "mixed_jam",
    "hamburger patty": "hamburger_patty",
    "chicken nugget": "chicken_nugget",
    # 複数形
    "noodles": "noodle",
    "green peas": "green_pea",
    "green_peas": "green_pea",
    "greenbeans": "green_beans",
    "green beans": "green_beans",
    "red beans": "red_beans",
    "soybean": "soybeans",
    "soy beans": "soybeans",
    # 日本語→英語（代表例）
    "ご飯":"rice","ごはん":"rice","ライス":"rice","白米":"rice",
    "パン":"bread","麺":"noodle","ヌードル":"noodle",
    "うどん":"udon","そば":"noodle","蕎麦":"noodle","ラーメン":"ramen_noodle",
    "スパゲッティ":"spaghetti","パスタ":"spaghetti","マカロニ":"macaroni",
    "じゃがいも":"potato","ポテト":"potato","さつまいも":"sweet_potato","里芋":"taro",
    "鶏肉":"chicken","チキン":"chicken","豚肉":"pork","ポーク":"pork","牛肉":"beef","ビーフ":"beef","レバー":"liver",
    "ししゃも":"shishamo","鮭":"salmon","サーモン":"salmon","鯖":"mackerel","サバ":"mackerel",
    "鯵":"horse_mackerel","アジ":"horse_mackerel","鰯":"sardine","イワシ":"sardine",
    "秋刀魚":"saury","サンマ":"saury","小魚":"mini_fish","しらす":"mini_fish","シラス":"mini_fish",
    "かえりちりめん":"kaeri_chirimen","ちりめん":"kaeri_chirimen",
    "ハム":"ham","ソーセージ":"sausage","ベーコン":"bacon",
    "かまぼこ":"kamaboko","ちくわ":"chikuwa",
    "豆腐":"tofu","納豆":"natto","油揚げ":"aburaage","厚揚げ":"aburaage","高野豆腐":"koya_tofu","豆乳":"soy_milk",
    "卵":"egg","玉子":"egg","うずら卵":"quail_egg",
    "牛乳":"milk","チーズ":"cheese","ヨーグルト":"yogurt",
    "ハンバーグ":"hamburger_patty","焼売":"shumai","シュウマイ":"shumai","餃子":"gyoza",
    "コロッケ":"croquette","ナゲット":"chicken_nugget","チキンナゲット":"chicken_nugget","ワンタン":"wonton",
    "キャベツ":"cabbage","白菜":"hakusai","小松菜":"komatsuna","ほうれん草":"spinach","レタス":"lettuce","水菜":"mizuna",
    "かぼちゃ":"pumpkin","南瓜":"pumpkin","なす":"eggplant","茄子":"eggplant",
    "ピーマン":"green_pepper","きゅうり":"cucumber","胡瓜":"cucumber","オクラ":"okra","ズッキーニ":"zucchini","トマト":"tomato",
    "にんじん":"carrot","人参":"carrot","大根":"radish","ごぼう":"burdock","牛蒡":"burdock","れんこん":"lotus_root","蓮根":"lotus_root",
    "玉ねぎ":"onion","たまねぎ":"onion","玉葱":"onion",
    "しいたけ":"shiitake","シイタケ":"shiitake","えのき":"enoki","エノキ":"enoki","しめじ":"shimeji","シメジ":"shimeji",
    "まいたけ":"maitake","舞茸":"maitake","エリンギ":"eryngii","なめこ":"nameko",
    "わかめ":"wakame","ひじき":"hijiki","昆布":"kombu",
    "枝豆":"edamame","グリーンピース":"green_pea","えんどう豆":"green_pea","そら豆":"broad_bean","ひよこ豆":"chickpea","レンズ豆":"lentil",
    "りんご":"apple","パイナップル":"pineapple","いちご":"strawberry","オレンジ":"orange","みかん":"orange",
    "ぶどう":"grape","バナナ":"banana","キウイ":"kiwi","メロン":"melon","梨":"pear","洋梨":"pear","桃":"peach",
    "コンポート":"fruit_compote","ミックスジャム":"mixed_jam","りんごピューレ":"apple_puree",
    "ブロッコリー":"broccoli","大豆":"soybeans","いんげん":"green_beans","インゲン":"green_beans","さやいんげん":"green_beans",
    "小豆":"red_beans","あずき":"red_beans","赤いんげん":"red_beans","金時豆":"red_beans",
    "漬物":"pickles","お漬物":"pickles","ねぎ":"green_onion","長ねぎ":"green_onion","青ねぎ":"green_onion","万能ねぎ":"green_onion",
    "アスパラ":"asparagus","ふき":"butterbur","蕗":"butterbur","はんぺん":"hanpen","もやし":"bean_sprout",
    "お好み焼き":"okonomiyaki","味噌汁":"miso_soup","しらたき":"shirataki","白滝":"shirataki","糸こんにゃく":"shirataki","きくらげ":"kikurage","木耳":"kikurage",
}

def load_label_set_from_cvat_xml(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = []
    for lab in root.findall(".//labels/label"):
        name_el = lab.find("./name")
        if name_el is not None and (name := (name_el.text or "").strip()):
            names.append(name)
    return sorted({n for n in names if n})

def normalize_label(name: str) -> str | None:
    n = (name or "").strip().lower()
    n = re.sub(r"[ \u3000\-]+", "_", n)  # 半/全角空白・ハイフン→アンダースコア
    n = SYNONYMS.get(n, n)
    return n if n in LABEL_SET else None

def extract_json_block(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else "{}"

def build_messages_pair(image_a: Path, image_b: Path) -> list[dict]:
    prompt = (
        "Compare the two images:\n"
        "- Image A shows the meal before eating.\n"
        "- Image B shows the same meal after eating.\n"
        "Identify which foods remain on the plates in Image B, based on the visible differences between Image A and Image B.\n"
        "Only respond with a JSON object exactly like: "
        "{\"remaining\": [<labels from this set only>]}.\n"
        f"Label set: {LABEL_SET}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_a), "disclaimer": "Image A (before)"},
            {"type": "image", "image": str(image_b), "disclaimer": "Image B (after)"},
            {"type": "text", "text": prompt},
        ],
    }]

@torch.inference_mode()
def infer_pair(model, processor, image_a: Path, image_b: Path,
               max_new_tokens: int, temperature: float) -> dict:
    messages = build_messages_pair(image_a, image_b)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
    )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    try:
        parsed = json.loads(extract_json_block(raw))
    except Exception:
        parsed = {"remaining": []}

    uniq: list[str] = []
    for name in parsed.get("remaining", []):
        n = normalize_label(name)
        if n and n not in uniq:
            uniq.append(n)

    return {"image_before": str(image_a), "image_after": str(image_b), "remaining": uniq, "raw": raw}

def read_pairs(csv_path: Path, before_col: str, after_col: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if before_col not in reader.fieldnames or after_col not in reader.fieldnames:
            raise AssertionError(f"CSV header must contain '{before_col}' and '{after_col}'. Found: {reader.fieldnames}")
        for row in reader:
            a = (row.get(before_col) or "").strip()
            b = (row.get(after_col) or "").strip()
            if not a or not b:
                continue
            pa, pb = Path(a), Path(b)
            pairs.append((pa, pb))
    return pairs

def main():
    global LABEL_SET

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=Path, required=True, help="before/after のペア一覧CSV")
    ap.add_argument("--before-col", default="before", help="CSVの食前列名（デフォルト: before）")
    ap.add_argument("--after-col",  default="after",  help="CSVの食後列名（デフォルト: after）")
    ap.add_argument("--xml", type=Path, required=True, help="CVAT XML（<labels>を含む）")

    # Qwen
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)

    # 出力
    ap.add_argument("--out-jsonl", type=Path, default=None, help="推論結果JSONL（デフォルト: pairs_csvとモデル名から自動）")
    ap.add_argument("--out-csv",   type=Path, default=None, help="推論結果CSV（任意）")

    args = ap.parse_args()

    # ラベル集合
    assert args.xml.exists(), f"Not found: {args.xml}"
    LABEL_SET = load_label_set_from_cvat_xml(args.xml)
    assert LABEL_SET, "[ERROR] <labels> が見つかりませんでした。XMLに<labels>が含まれているか確認してください。"
    print(f"[INFO] Loaded {len(LABEL_SET)} labels from {args.xml.name}")

    # ペア読み込み
    assert args.pairs_csv.exists(), f"Not found: {args.pairs_csv}"
    pairs = read_pairs(args.pairs_csv, args.before_col, args.after_col)
    print(f"[INFO] Loaded pairs: {len(pairs)}")

    if not pairs:
        print("[WARN] No pairs to run.")
        sys.exit(0)

    # DTYPE
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        torch_dtype = None  # let HF decide with device_map
    else:
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map[args.dtype]
        torch_dtype = dtype

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # 出力パス
    if args.out_jsonl is None:
        name = args.model.split("/")[-1]
        args.out_jsonl = args.pairs_csv.with_name(f"results_remaining_pairs__{args.pairs_csv.stem}__{name}.jsonl")
    if args.out_csv is None:
        args.out_csv = args.pairs_csv.with_name(f"results_remaining_pairs__{args.pairs_csv.stem}.csv")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 推論
    done = 0
    with args.out_jsonl.open("w", encoding="utf-8") as fj, args.out_csv.open("w", encoding="utf-8-sig", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["before", "after", "remaining_json", "remaining_joined", "raw"])
        for i, (pa, pb) in enumerate(pairs, 1):
            try:
                if not pa.exists() or not pb.exists():
                    res = {
                        "image_before": str(pa),
                        "image_after": str(pb),
                        "remaining": [],
                        "error": "file_not_found" if (not pa.exists() or not pb.exists()) else None
                    }
                else:
                    res = infer_pair(model, processor, pa, pb, args.max_new_tokens, args.temperature)
            except Exception as e:
                res = {"image_before": str(pa), "image_after": str(pb), "remaining": [], "error": repr(e)}

            fj.write(json.dumps(res, ensure_ascii=False) + "\n")
            w.writerow([
                res["image_before"],
                res["image_after"],
                json.dumps(res.get("remaining", []), ensure_ascii=False),
                ";".join(res.get("remaining", [])),
                res.get("raw","")
            ])
            done += 1
            if (i % 10 == 0) or (i == len(pairs)):
                print(f"[{i}/{len(pairs)}] {pa.name} | {pb.name} -> {res.get('remaining')}")

    print(f"[DONE] {done} pairs -> {args.out_jsonl} , {args.out_csv}")

if __name__ == "__main__":
    main()
