# run_qwen_vl_remaining.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # Qwen付属のユーティリティ
import time

# ====== ラベル集合はXMLから動的ロード ======
LABEL_SET: list[str] = []  # main() でロードして上書き

# ==== 軽量シノニム（XMLの英語名を壊さない最小限） ====
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
    # 追加した食材（例）
    "お好み焼き":"okonomiyaki","味噌汁":"miso_soup","しらたき":"shirataki","白滝":"shirataki","糸こんにゃく":"shirataki","きくらげ":"kikurage","木耳":"kikurage",
}

def load_label_set_from_cvat_xml(xml_path: Path) -> list[str]:
    """CVAT XML (<labels><label><name>) からラベル名を抽出して昇順で返す。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = []
    for lab in root.findall(".//labels/label"):
        name_el = lab.find("./name")
        if name_el is not None and (name := (name_el.text or "").strip()):
            names.append(name)
    uniq = sorted({n for n in names if n})
    return uniq

def normalize_label(name: str) -> str | None:
    n = (name or "").strip().lower()
    n = re.sub(r"[ \u3000\-]+", "_", n)  # 半/全角空白・ハイフン→アンダースコア
    n = SYNONYMS.get(n, n)
    return n if n in LABEL_SET else None

def extract_json_block(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else "{}"

def build_messages(image_path: Path) -> list[dict]:
    prompt = (
        "Identify which foods remain on the plate in the image. "
        "Only respond with a JSON object exactly like: "
        "{\"remaining\": [<labels from this set only>]}.\n"
        f"Label set: {LABEL_SET}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]

def infer_one(model, processor, image_path: Path, max_new_tokens: int, temperature: float) -> dict:
    messages = build_messages(image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.inference_mode():
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

    return {"image": str(image_path), "remaining": uniq, "raw": raw}

# ========== CVAT XML helpers ==========
def collect_image_entries(root: ET.Element):
    return root.findall(".//image")

def labels_in_image(img_elem: ET.Element) -> set[str]:
    labs = set()
    for t in img_elem.findall("./tag"):
        lab = normalize_label(t.get("label"))
        if lab: labs.add(lab)
    for shape in img_elem:
        if isinstance(shape.tag, str) and shape.get("label"):
            lab = normalize_label(shape.get("label"))
            if lab: labs.add(lab)
    return labs

def number_in_stem(pathlike: str) -> int | None:
    m = re.search(r"(\d+)", Path(pathlike).stem)
    return int(m.group(1)) if m else None

def get_images_from_xml(xml: Path, images_root: Path,
                        include: set[str], exclude: set[str],
                        range_ids: range | None, pattern: str | None) -> list[Path]:
    tree = ET.parse(xml)
    root = tree.getroot()
    imgs = collect_image_entries(root)
    out = []
    for img in imgs:
        name = img.get("name")
        if not name:
            continue
        if pattern and (pattern not in name):
            continue
        if range_ids is not None:
            num = number_in_stem(name)
            if (num is None) or (num not in range_ids):
                continue
        labs = labels_in_image(img)
        ok_inc = (not include) or (labs & include)
        ok_exc = not (labs & exclude)
        if not (ok_inc and ok_exc):
            continue
        src = (images_root / name).resolve()
        if not src.exists():
            alt = Path(name)
            if alt.is_absolute() and alt.exists():
                src = alt
            else:
                print(f"[WARN] missing image: {src}")
                continue
        out.append(src)
    return sorted(out)

def get_images_from_dir(after_dir: Path, pattern: str | None, range_ids: range | None) -> list[Path]:
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    imgs = [p for p in after_dir.rglob("*") if p.suffix.lower() in exts]
    if pattern:
        imgs = [p for p in imgs if pattern in p.name]
    if range_ids is not None:
        imgs = [p for p in imgs if (number_in_stem(p.name) in range_ids)]
    return sorted(imgs)

# ========== main ==========
def main():
    global LABEL_SET

    ap = argparse.ArgumentParser()
    # 相互排他（--after_dir か --xml のどちらか必須）
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--after_dir", type=Path, help="ディレクトリ直下/配下の画像を再帰で収集")
    src.add_argument("--xml", type=Path, help="CVAT XML（<labels> と <image> を含む）")
    ap.add_argument("--images-root", type=Path, help="--xml使用時、XMLのnameと結合する画像ルート")

    # フィルタ
    ap.add_argument("--pattern", default=None, help="ファイル名の部分一致フィルタ")
    ap.add_argument("--range", default="", help="番号範囲（例: img10-30 / 10-30 どちらでも可）")
    ap.add_argument("--include", default="", help="含めるラベル（カンマ区切り）")
    ap.add_argument("--exclude", default="", help="除外するラベル（カンマ区切り）")

    # 推論
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=1)

    # 出力
    ap.add_argument("--out", type=Path, default=None, help="保存先 .jsonl（デフォルトは自動生成）")
    args = ap.parse_args()

    # LABEL_SET を --xml からロード
    if args.xml:
        assert args.xml.exists(), f"Not found: {args.xml}"
        LABEL_SET = load_label_set_from_cvat_xml(args.xml)
        assert LABEL_SET, "[ERROR] <labels> が見つかりませんでした。XMLに<labels>が含まれているか確認してください。"
        print(f"[INFO] Loaded {len(LABEL_SET)} labels from {args.xml.name}")
    else:
        print("[WARN] --after_dir モードでは LABEL_SET をXMLから自動取得できません。現行SYNONYMSに合うクラスのみ正規化されます。")

    # range 解析
    range_ids = None
    if args.range:
        m = re.search(r"(\d+)\s*-\s*(\d+)", args.range)
        if m:
            s, e = map(int, m.groups())
            range_ids = range(s, e + 1)
            print(f"[INFO] Filtering by range: {s}–{e}")

    # include/exclude 正規化
    def parse_labels(s: str) -> set[str]:
        labs = set()
        for x in filter(None, [t.strip() for t in s.split(",")]):
            nx = normalize_label(x)
            if nx: labs.add(nx)
        return labs
    include = parse_labels(args.include)
    exclude = parse_labels(args.exclude)

    # 画像列挙
    if args.xml:
        assert args.images_root, "--xml を使う場合は --images-root も指定してください"
        assert args.images_root.exists(), f"Not found: {args.images_root}"
        imgs = get_images_from_xml(args.xml, args.images_root, include, exclude, range_ids, args.pattern)
    else:
        assert args.after_dir and args.after_dir.exists(), f"Not found: {args.after_dir}"
        imgs = get_images_from_dir(args.after_dir, args.pattern, range_ids)

    print(f"[INFO] Found {len(imgs)} images")

    # DTYPE
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # モデル読み込み（auto でも dtype を渡すように変更）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,              # ← ここを固定で dtype に
        device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # 出力パス設定（あなたのコードのままでOK）
    if args.out is None:
        base = args.after_dir.name if args.after_dir else Path(args.images_root).name
        name = args.model.split("/")[-1]
        args.out = Path(".") / f"results_remaining_{base}_{name}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ==== 推論速度計測 ====
    # 最初の数枚はウォームアップとして統計から除外（GPU初回呼び出しのオーバヘッド回避）
    WARMUP = 3
    durations = []  # (秒) の配列。ウォームアップ後のみ格納
    overall_start = time.perf_counter()

    done = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, path in enumerate(imgs, 1):
            try:
                # 計測開始
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                res = infer_one(model, processor, path, args.max_new_tokens, args.temperature)

                # 計測終了
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                if i > WARMUP:
                    durations.append(t1 - t0)

            except Exception as e:
                res = {"image": str(path), "remaining": [], "error": repr(e)}

            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            done += 1
            if (i % 10 == 0) or (i == len(imgs)):
                print(f"[{i}/{len(imgs)}] {path.name} -> {res.get('remaining')}")

    overall_end = time.perf_counter()
    total_time = overall_end - overall_start

    # ==== 集計（ウォームアップ除外後の durations を使用）====
    def percentile(vals, p):
        if not vals:
            return 0.0
        s = sorted(vals)
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[k]

    n_timed = len(durations)
    avg_sec = sum(durations) / n_timed if n_timed > 0 else 0.0
    img_per_sec = (n_timed / sum(durations)) if n_timed > 0 and sum(durations) > 0 else 0.0
    p50 = percentile(durations, 50)
    p95 = percentile(durations, 95)

    stats = {
        "total_images": len(imgs),
        "timed_images": n_timed,
        "warmup_images": min(WARMUP, len(imgs)),
        "overall_wall_time_sec": total_time,
        "avg_latency_per_image_ms": avg_sec * 1000.0,
        "throughput_images_per_sec": img_per_sec,
        "p50_latency_ms": p50 * 1000.0,
        "p95_latency_ms": p95 * 1000.0,
        "model": args.model,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "source_mode": "xml" if args.xml else "after_dir",
    }

    # 画面に表示
    print("[SPEED]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 結果の隣に .stats.json を保存
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {done} samples -> {args.out}")
    print(f"[OK] Wrote speed stats -> {stats_path}")

if __name__ == "__main__":
    main()
