# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # Qwen 付属

# =========================
# ① 同義語（→ 細粒度名に正規化）
# =========================
# ここは「細粒度ラベル」に寄せる辞書です（英語・日本語・ゆらぎ表記など）
SYN_TO_FINE = {
    # ---- 空白/ハイフン変種 → 細粒度 ----
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

    # ---- 複数形 → 単数（細粒度） ----
    "noodles": "noodle",
    "green peas": "green_pea",
    "green_peas": "green_pea",
    "green beans": "green_beans",
    "greenbeans": "green_beans",
    "red beans": "red_beans",
    "soy beans": "soybeans",
    "soybean": "soybeans",

    # ---- 一般名 → 具体（細粒度） ----
    "pasta": "spaghetti",
    "ramen": "ramen_noodle",

    # ---- 日本語（→ 細粒度） ----
    "ご飯": "rice","ごはん": "rice","ライス": "rice","白米": "rice",
    "パン": "bread",
    "麺": "noodle","ヌードル": "noodle",
    "うどん": "udon","そば": "noodle","蕎麦": "noodle",
    "ラーメン": "ramen_noodle",
    "スパゲッティ": "spaghetti","パスタ": "spaghetti","マカロニ": "macaroni",
    "じゃがいも": "potato","ポテト": "potato",
    "さつまいも": "sweet_potato","里芋": "taro",
    "鶏肉": "chicken","チキン": "chicken",
    "豚肉": "pork","ポーク": "pork",
    "牛肉": "beef","ビーフ": "beef",
    "レバー": "liver",
    "ししゃも": "shishamo",
    "鮭": "salmon","サーモン": "salmon",
    "鯖": "mackerel","サバ": "mackerel",
    "鯵": "horse_mackerel","アジ": "horse_mackerel",
    "鰯": "sardine","イワシ": "sardine",
    "秋刀魚": "saury","サンマ": "saury",
    "小魚": "mini_fish","しらす": "mini_fish","シラス": "mini_fish",
    "かえりちりめん": "kaeri_chirimen","ちりめん": "kaeri_chirimen",
    "ハム": "ham","ソーセージ": "sausage","ベーコン": "bacon",
    "かまぼこ": "kamaboko","ちくわ": "chikuwa",
    "豆腐": "tofu","納豆": "natto","油揚げ": "aburaage","厚揚げ": "aburaage",
    "高野豆腐": "koya_tofu","凍り豆腐": "koya_tofu","豆乳": "soy_milk",
    "卵": "egg","玉子": "egg","うずら卵": "quail_egg",
    "牛乳": "milk","チーズ": "cheese","ヨーグルト": "yogurt",
    "ハンバーグ": "hamburger_patty","ハンバーガーパティ": "hamburger_patty",
    "焼売": "shumai","シュウマイ": "shumai",
    "餃子": "gyoza",
    "コロッケ": "croquette",
    "ナゲット": "chicken_nugget","チキンナゲット": "chicken_nugget",
    "ワンタン": "wonton",
    "キャベツ": "cabbage","白菜": "hakusai","小松菜": "komatsuna",
    "ほうれん草": "spinach","レタス": "lettuce","水菜": "mizuna",
    "かぼちゃ": "pumpkin","南瓜": "pumpkin","なす": "eggplant","茄子": "eggplant",
    "ピーマン": "green_pepper",
    "きゅうり": "cucumber","胡瓜": "cucumber",
    "オクラ": "okra","ズッキーニ": "zucchini","トマト": "tomato",
    "にんじん": "carrot","人参": "carrot",
    "大根": "radish","ごぼう": "burdock","牛蒡": "burdock",
    "れんこん": "lotus_root","蓮根": "lotus_root",
    "玉ねぎ": "onion","たまねぎ": "onion","玉葱": "onion",
    "しいたけ": "shiitake","シイタケ": "shiitake",
    "えのき": "enoki","エノキ": "enoki",
    "しめじ": "shimeji","シメジ": "shimeji",
    "まいたけ": "maitake","舞茸": "maitake",
    "エリンギ": "eryngii","なめこ": "nameko",
    "わかめ": "wakame","ひじき": "hijiki","昆布": "kombu",
    "枝豆": "edamame","グリーンピース": "green_pea","えんどう豆": "green_pea",
    "そら豆": "broad_bean","ひよこ豆": "chickpea","レンズ豆": "lentil",
    "りんご": "apple","パイナップル": "pineapple","いちご": "strawberry",
    "オレンジ": "orange","みかん": "orange",
    "ぶどう": "grape","バナナ": "banana","キウイ": "kiwi",
    "メロン": "melon","梨": "pear","洋梨": "pear","桃": "peach",
    "コンポート": "fruit_compote","ミックスジャム": "mixed_jam","りんごピューレ": "apple_puree",
    "ブロッコリー": "broccoli",
    "大豆": "soybeans",
    "いんげん": "green_beans","インゲン": "green_beans","さやいんげん": "green_beans",
    "小豆": "red_beans","あずき": "red_beans","赤いんげん": "red_beans","金時豆": "red_beans",
    "漬物": "pickles","お漬物": "pickles",
    "ねぎ": "green_onion","長ねぎ": "green_onion","青ねぎ": "green_onion","万能ねぎ": "green_onion",
    "アスパラ": "asparagus","ふき": "butterbur","蕗": "butterbur",
    "はんぺん": "hanpen",
    "もやし": "bean_sprout",
}

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[ \u3000\-]+", "_", s)
    return s

def canonicalize_fine(name: str) -> str:
    """ゆらぎを正規化して細粒度候補名に寄せる"""
    n = _slug(name)
    return SYN_TO_FINE.get(n, n)

# ======================================================
# ② XML から「細粒度 → subclass」と「subclass 集合」を抽出
# ======================================================
def fine_to_subclass_from_xml(xml_path: Path) -> tuple[dict[str, str], list[str]]:
    """
    CVAT XML（images 1.x）から：
      - 各 <label> の中の <attribute><name>subclass</name><values>XXXX</values>
        を読み取り、細粒度ラベル（<label><name>）→ subclass の対応を作る
      - subclass 値の全集合を返す
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fine2sub: dict[str, str] = {}
    subclasses: set[str] = set()

    for lab in root.findall(".//labels/label"):
        fine = lab.findtext("name") or ""
        fine = canonicalize_fine(fine)
        if not fine:
            continue

        # この label に定義されている attribute 群を走査
        subval = None
        for attr in lab.findall("./attributes/attribute"):
            aname = (attr.findtext("name") or "").strip().lower()
            if aname == "subclass":
                val = attr.findtext("values") or ""
                val = _slug(val)
                if val:
                    subval = val
                    subclasses.add(val)
                    break

        if subval:
            fine2sub[fine] = subval

    return fine2sub, sorted(subclasses)

# ======================================================
# ③ 画像列挙（XML or ディレクトリ）
# ======================================================
def number_in_stem(pathlike: str) -> int | None:
    m = re.search(r"(\d+)", Path(pathlike).stem)
    return int(m.group(1)) if m else None

def collect_images_from_xml(xml: Path, images_root: Path,
                            pattern: str | None, range_ids: range | None) -> list[Path]:
    tree = ET.parse(xml)
    root = tree.getroot()
    out = []
    for img in root.findall(".//image"):
        name = img.get("name")
        if not name:
            continue
        if pattern and (pattern not in name):
            continue
        if range_ids is not None:
            num = number_in_stem(name)
            if (num is None) or (num not in range_ids):
                continue
        p = (images_root / name).resolve()
        if p.exists():
            out.append(p)
        else:
            alt = Path(name)
            if alt.is_absolute() and alt.exists():
                out.append(alt)
            else:
                print(f"[WARN] missing image: {p}")
    return sorted(out)

def collect_images_from_dir(after_dir: Path, pattern: str | None, range_ids: range | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in after_dir.rglob("*") if p.suffix.lower() in exts]
    if pattern:
        imgs = [p for p in imgs if pattern in p.name]
    if range_ids is not None:
        imgs = [p for p in imgs if (number_in_stem(p.name) in range_ids)]
    return sorted(imgs)

# ======================================================
# ④ VLM 推論 → 正規化 → subclass へ集約
# ======================================================
def build_messages(image_path: Path, subclass_label_set: list[str]) -> list[dict]:
    prompt = (
        "Identify which foods remain on the plate in the image.\n"
        "Return ONLY a JSON object like:\n"
        "{\"remaining\": [<labels from this set only>]} \n"
        "Use these labels (coarse categories):\n"
        f"{subclass_label_set}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]

def vlm_infer_one(model, processor, image_path: Path,
                  subclass_label_set: list[str],
                  max_new_tokens: int, temperature: float) -> dict:
    messages = build_messages(image_path, subclass_label_set)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # JSON 抜き出し（壊れにくい簡易パーサ）
    m = re.search(r"\{.*\}", raw, flags=re.S)
    js = {}
    if m:
        try:
            js = json.loads(m.group(0))
        except Exception:
            js = {}
    remain = js.get("remaining", []) if isinstance(js, dict) else []

    return {"image": str(image_path), "remaining_raw": remain, "raw": raw}

def normalize_and_map_to_subclass(
    items: list[str],
    fine2sub: dict[str, str],
    subclass_label_set: set[str]
) -> list[str]:
    """
    1) 同義語で細粒度に正規化
    2) 細粒度→subclass に置換
    3) もともと subclass 名が来た場合はそのまま許容
    """
    out: list[str] = []
    for it in items or []:
        # まず「細粒度」へ正規化（失敗なら素のスラッグ化）
        fine = canonicalize_fine(it)
        # 細粒度 → subclass へ
        sub = fine2sub.get(fine)
        if sub is None:
            # もともと subclass を直接返してくる場合も許容
            maybe_sub = _slug(it)
            if maybe_sub in subclass_label_set:
                sub = maybe_sub

        if sub and (sub not in out):
            out.append(sub)

    return out

# ======================================================
# ⑤ main
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--after_dir", type=Path, help="画像ディレクトリ（再帰）")
    src.add_argument("--xml", type=Path, help="CVAT XML (images 1.x)")
    ap.add_argument("--images-root", type=Path, help="--xml 使用時に画像のルートを指定")

    ap.add_argument("--pattern", default=None)
    ap.add_argument("--range", dest="range_text", default="", help="例: 10-30")

    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--out", type=Path, default=Path("./results_remaining_subclass.jsonl"))
    args = ap.parse_args()

    # 範囲
    range_ids = None
    if args.range_text:
        m = re.search(r"(\d+)\s*-\s*(\d+)", args.range_text)
        if m:
            s, e = map(int, m.groups())
            range_ids = range(s, e + 1)
            print(f"[INFO] Filter by range: {s}-{e}")

    # 画像列挙
    if args.xml:
        assert args.images_root and args.images_root.exists(), "--xml 使用時は --images-root 必須"
        imgs = collect_images_from_xml(args.xml, args.images_root, args.pattern, range_ids)
    else:
        assert args.after_dir and args.after_dir.exists(), "Not found: --after_dir"
        imgs = collect_images_from_dir(args.after_dir, args.pattern, range_ids)

    if not imgs:
        print("[WARN] No images found.")
        return

    # XML から「細粒度→subclass」「subclass 集合」を抽出
    xml_for_schema = args.xml
    if not xml_for_schema and args.after_dir:
        # after_dir だけの場合でも、同ディレクトリに annotations.xml があれば拾う
        cand = args.after_dir.parent / "annotations.xml"
        if cand.exists():
            xml_for_schema = cand

    if not xml_for_schema:
        raise SystemExit("[ERROR] subclass ラベル集合とマッピングを抽出するために XML が必要です。 --xml を指定してください。")

    fine2sub, subclass_list = fine_to_subclass_from_xml(xml_for_schema)
    subclass_set = set(subclass_list)
    print(f"[INFO] subclass labels ({len(subclass_list)}): {subclass_list}")

    # モデル読み込み
    if args.dtype == "auto":
        torch_dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                       else torch.float16)
    else:
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=None if args.dtype == "auto" else torch_dtype,
        device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # 推論
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(imgs, 1):
            try:
                raw_res = vlm_infer_one(
                    model, processor, p, subclass_list,
                    args.max_new_tokens, args.temperature
                )
                mapped = normalize_and_map_to_subclass(
                    raw_res.get("remaining_raw", []),
                    fine2sub=fine2sub,
                    subclass_label_set=subclass_set,
                )
                out = {"image": str(p), "remaining": mapped, "raw": raw_res.get("raw", "")}
            except Exception as e:
                out = {"image": str(p), "remaining": [], "error": repr(e)}

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            done += 1
            if (i % 10 == 0) or (i == len(imgs)):
                print(f"[{i}/{len(imgs)}] {p.name} -> {out.get('remaining')}")
    print(f"[DONE] {done} samples -> {args.out}")

if __name__ == "__main__":
    main()
