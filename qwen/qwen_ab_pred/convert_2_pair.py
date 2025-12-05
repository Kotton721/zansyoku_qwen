# convert_2_pair.py  (encoding auto-detect & robust CSV/JSON loader)
from __future__ import annotations
import argparse, csv, json, sys, io
from pathlib import Path
import pandas as pd

# ---- Encoding helpers -------------------------------------------------
_CANDIDATE_ENCODINGS = [
    "utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp", "windows-1252", "latin1",
]

def open_text_smart(path: Path) -> io.StringIO:
    """Try multiple encodings and return a text buffer for generic readers."""
    for enc in _CANDIDATE_ENCODINGS:
        try:
            data = path.read_bytes().decode(enc, errors="strict")
            return io.StringIO(data)
        except UnicodeDecodeError:
            continue
    # 最後の手段：破損文字を置換してでも読む
    data = path.read_bytes().decode(_CANDIDATE_ENCODINGS[0], errors="replace")
    return io.StringIO(data)

def read_csv_smart(path: Path) -> pd.DataFrame:
    """Use pandas with auto sep sniffing and multiple encodings."""
    for enc in _CANDIDATE_ENCODINGS:
        try:
            # engine='python' で区切り文字の自動判定(sep=None)が使える
            return pd.read_csv(path, encoding=enc, engine="python", sep=None)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # 解析失敗（例：区切り文字混在など）は次に回す
            last_err = e
    # 最後の手段：エンコードは通した上で標準カンマ区切りで読む
    buf = open_text_smart(path)
    try:
        return pd.read_csv(buf, engine="python")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {path} ({e})", file=sys.stderr)
        raise

# ---- Path & mapping helpers ------------------------------------------
def normalize_old_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace(" ", "").replace("\u3000", "")
    return Path(s).name.lower()

def _common_path(a: Path, b: Path) -> str:
    a_parts = a.resolve().parts
    b_parts = b.resolve().parts
    i = 0
    while i < min(len(a_parts), len(b_parts)) and a_parts[i] == b_parts[i]:
        i += 1
    return str(Path(*a_parts[:i])) if i else "/"

def _guess_common_root(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    common = Path(paths[0])
    for p in paths[1:]:
        common = Path(_common_path(common, p))
        if str(common) == "/":
            break
    return common

def rel_to_target(new_path: Path, old_root: Path | None, target_root: Path | None) -> Path:
    if old_root is None or target_root is None:
        return new_path
    try:
        rel = new_path.relative_to(old_root)
        return target_root / rel
    except Exception:
        parent_name = new_path.parent.name
        if parent_name in ("before", "after"):
            return target_root / parent_name / new_path.name
        return new_path

def load_mapping(mapping_path: Path):
    """
    Return:
      - by_old: dict[old_name(normalized) -> Path(new_path)]
      - guessed_old_root: Path | None
    CSV: needs header with at least old_name, new_path
    JSON: list of objects with old_name/new_path
    """
    by_old: dict[str, Path] = {}
    rows = []
    if mapping_path.suffix.lower() == ".csv":
        buf = open_text_smart(mapping_path)
        rows = list(csv.DictReader(buf))
    elif mapping_path.suffix.lower() == ".json":
        buf = open_text_smart(mapping_path)
        rows = json.loads(buf.read())
        if not isinstance(rows, list):
            raise ValueError("JSON mapping must be a list of objects.")
    else:
        raise ValueError("mapping must be .csv or .json")

    new_paths = []
    for r in rows:
        old_name = normalize_old_name(r.get("old_name", ""))
        np_str = r.get("new_path") or r.get("newname") or r.get("new") or r.get("new_file") or ""
        if not old_name or not np_str:
            continue
        p = Path(np_str)
        by_old[old_name] = p
        new_paths.append(p)

    guessed_old_root = _guess_common_root(new_paths) if new_paths else None
    return by_old, guessed_old_root

# ---- Main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="旧名ペアCSV（例: indi_data.csv）")
    ap.add_argument("--mapping", required=True, type=Path, help="旧→新対応（filename_map.csv/json）")
    ap.add_argument("--old-root", type=Path, default=None, help="mapping内 new_path の現ルート（省略時は推定）")
    ap.add_argument("--target-root", type=Path, default=None, help="出力で付け替えるルート")
    ap.add_argument("--out-csv", type=Path, default=Path("pairs_new.csv"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("pairs_new.jsonl"))
    args = ap.parse_args()

    # mapping
    by_old, guessed_old_root = load_mapping(args.mapping)
    if not by_old:
        print("[ERROR] mapping が空です。old_name/new_path を確認してください。", file=sys.stderr)
        sys.exit(1)

    old_root = args.old_root or guessed_old_root
    if old_root:
        print(f"[INFO] old_root: {old_root}")
    if args.target_root:
        print(f"[INFO] target_root: {args.target_root}")

    # CSV（エンコード自動）
    df = read_csv_smart(args.csv)

    # 列名候補
    COL_BEFORE_CAND = ["食前画像", "pre_image", "before_image"]
    COL_AFTER_CAND  = ["食後画像", "post_image", "after_image"]
    col_before = next((c for c in COL_BEFORE_CAND if c in df.columns), None)
    col_after  = next((c for c in COL_AFTER_CAND  if c in df.columns), None)
    if not col_before or not col_after:
        print(f"[ERROR] 必要列が見つかりません: {COL_BEFORE_CAND} / {COL_AFTER_CAND}", file=sys.stderr)
        sys.exit(1)

    pairs_rows, pairs_jsonl = [], []
    miss_bef = miss_aft = 0

    for _, row in df.iterrows():
        old_b = normalize_old_name(row[col_before])
        old_a = normalize_old_name(row[col_after])

        new_b = by_old.get(old_b)
        new_a = by_old.get(old_a)

        if new_b is None: miss_bef += 1
        if new_a is None: miss_aft += 1

        if new_b is not None:
            new_b = rel_to_target(new_b, old_root, args.target_root)
        if new_a is not None:
            new_a = rel_to_target(new_a, old_root, args.target_root)

        pairs_rows.append({
            "before_new": "" if new_b is None else str(new_b),
            "after_new":  "" if new_a is None else str(new_a),
        })
        pairs_jsonl.append({
            "before_new": None if new_b is None else str(new_b),
            "after_new":  None if new_a is None else str(new_a),
        })

    pd.DataFrame(pairs_rows).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for obj in pairs_jsonl:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[DONE] pairs -> {args.out_csv} , {args.out_jsonl}")
    print(f"[INFO] missing: before={miss_bef}, after={miss_aft}")

if __name__ == "__main__":
    main()

