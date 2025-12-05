from pathlib import Path
import re
import shutil
import csv
import json

# =========================
# 設定
# =========================
ROOT = Path("/home/user/qwen/qwen_ab_pred/dataset/indi_image/images")  # 元フォルダ
OUT_BEFORE = ROOT / "before"
OUT_AFTER  = ROOT / "after"

DRY_RUN = False         # 最初は True で動作確認。OKなら False にして実行
COPY_MODE = False      # True: コピーして元を残す / False: 移動（rename）

# 出力されるマッピングファイル
MAP_CSV  = ROOT / "filename_map.csv"
MAP_JSON = ROOT / "filename_map.json"

# =========================
# 定義
# =========================

# 対応拡張子
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 例: [画像1-2] _230421_0_num0_0.7909.jpg
#      └──gid──┘ └p┘  └date┘ └idx┘ └─num──┘ └score┘   └ext┘
PATTERN = re.compile(
    r"""
    ^\[
        (?P<gid>.+?)            # 画像グループID（例: 画像1 / img1 / 写真12 など）
        -
        (?P<phase>[12])         # 1=before, 2=after
    \]
    \s*_?
        (?P<date>\d{6,8})       # 230421 など（6～8桁）
        _
        (?P<idx>\d+)            # 料理インデックス（任意の連番）
        _
        (?P<numtag>num\d+)      # num0 など
        _
        (?P<score>\d+(?:\.\d+)?)# 0.7909 など（使わないが吸収）
    \.(?P<ext>jpg|jpeg|png|webp|bmp)$
    """,
    re.IGNORECASE | re.VERBOSE
)

# 全角→半角（数字）用
_Z2H = str.maketrans("０１２３４５６７８９", "0123456789")

def ensure_dirs():
    OUT_BEFORE.mkdir(parents=True, exist_ok=True)
    OUT_AFTER.mkdir(parents=True, exist_ok=True)

def normalize_gid(gid: str) -> str:
    """
    例:
      '画像1'      -> 'img1'
      '  画像１２' -> 'img12'
      '写真3'      -> 'img3'
      'img_4'     -> 'img4'
    """
    s = (gid or "").strip()
    s = s.translate(_Z2H)  # 全角数字→半角
    # よくある日本語プレフィックスを 'img' に正規化
    s = re.sub(r"^(画像|寫真|写真)\s*", "img", s, flags=re.IGNORECASE)
    # 先頭が英字でない場合は 'img' を補う（安全側）
    if not re.match(r"^[A-Za-z]", s):
        s = "img" + s
    # 許容文字以外を除去（英数字とアンダースコア以外は消す）
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = s.lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "img"

def parse_name(path: Path):
    """
    ファイル名をパターン解析して正規化情報を返す。
    マッチしなければ None。
    """
    # スペースの混入（例: "] _"）への耐性を上げる
    name_for_match = path.name.replace(" ", "")
    m = PATTERN.match(name_for_match)
    if not m:
        return None
    d = m.groupdict()
    gid = normalize_gid(d["gid"])
    phase = "before" if d["phase"] == "1" else "after"
    info = {
        "gid": gid,                 # 正規化後のグループ（例: img1）
        "phase": phase,             # "before" or "after"
        "date": d["date"],          # 230421 など
        "idx": d["idx"],            # 0 など
        "numtag": d["numtag"],      # num0 など（新名には使わない）
        "ext": "." + d["ext"].lower()
    }
    return info

def build_new_name(info: dict) -> str:
    """
    新ファイル名を構成。
    例: img1_230421_0_before.jpg
    """
    return f"{info['gid']}_{info['date']}_{info['idx']}_{info['phase']}{info['ext']}"

def unique_destination(dst_dir: Path, base_name: str) -> Path:
    """
    既存衝突を避けるため、必要なら __1, __2 を付与してユニークなパスを返す。
    """
    dst = dst_dir / base_name
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    k = 1
    while (dst_dir / f"{stem}__{k}{ext}").exists():
        k += 1
    return dst_dir / f"{stem}__{k}{ext}"

def save_mapping(mapping_rows: list[dict]):
    """
    mapping_rows を CSV と JSON に保存。
    """
    # CSV
    with MAP_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["old_path", "new_path", "old_name", "new_name", "phase", "action", "reason"]
        )
        writer.writeheader()
        writer.writerows(mapping_rows)
    # JSON
    with MAP_JSON.open("w", encoding="utf-8") as f:
        json.dump(mapping_rows, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Mapping saved: {MAP_CSV}")
    print(f"[DONE] Mapping saved: {MAP_JSON}")

def main():
    assert ROOT.exists(), f"Not found: {ROOT}"
    ensure_dirs()

    # すでに before/after に入っているものは対象外（重複処理防止）
    candidates = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
        and (OUT_BEFORE not in p.parents) and (OUT_AFTER not in p.parents)
    ]

    print(f"[INFO] Found {len(candidates)} files to consider under: {ROOT}")

    moved = 0
    unmatched = []

    mapping_rows = []  # 対応表レコード

    for p in candidates:
        info = parse_name(p)
        if not info:
            unmatched.append(p)
            mapping_rows.append({
                "old_path": str(p),
                "new_path": "",
                "old_name": p.name,
                "new_name": "",
                "phase": "",
                "action": "skip",
                "reason": "unmatched_pattern"
            })
            continue

        new_name = build_new_name(info)
        dst_dir = OUT_BEFORE if info["phase"] == "before" else OUT_AFTER
        dst = unique_destination(dst_dir, new_name)

        rel_dst = str(dst.relative_to(ROOT))
        action = "dry_run"
        reason = ""

        if not DRY_RUN:
            if COPY_MODE:
                shutil.copy2(str(p), str(dst))
                action = "copied"
            else:
                shutil.move(str(p), str(dst))
                action = "moved"
            moved += 1

        print(f"[{action.upper()}] {p.name} -> {rel_dst}")

        mapping_rows.append({
            "old_path": str(p),
            "new_path": str(dst),
            "old_name": p.name,
            "new_name": dst.name,
            "phase": info["phase"],
            "action": action,
            "reason": reason
        })

    # 結果保存（DRYでも保存して確認できるようにする）
    save_mapping(mapping_rows)

    # レポート
    print("\n=== REPORT ===")
    print(f"Candidates: {len(candidates)}")
    print(f"Moved/Copied (when DRY_RUN=False): {moved}")
    print(f"Unmatched pattern: {len(unmatched)}")
    if unmatched:
        print("Examples of unmatched:")
        for up in unmatched[:10]:
            print("  -", up.name)

if __name__ == "__main__":
    main()

