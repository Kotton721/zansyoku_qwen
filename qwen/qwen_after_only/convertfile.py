from pathlib import Path
import re, shutil

# === 設定 ===
ROOT = Path("/home/user/qwen/datasets/indi_image/images")  # 元フォルダ
OUT_BEFORE = ROOT / "before"
OUT_AFTER  = ROOT / "after"
DRY_RUN = True  # ← 最初は True で動作確認。OKなら False にして実行。


# 対応拡張子
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 例: [画像1-2] _230421_0_num0_0.7909.jpg
#      └──gid──┘ └p┘  └date┘ └idx┘ └─num──┘ └score┘   └ext┘
PATTERN = re.compile(
    r"""
    ^\[
        (?P<gid>.+?)       # 画像グループID（例: 画像1）
        -
        (?P<phase>[12])    # 1=before, 2=after
    \]
    \s*_?
        (?P<date>\d{6,8})  # 230421 など
        _
        (?P<idx>\d+)       # 0 など（任意の連番）
        _
        (?P<numtag>num\d+) # num0 など
        _
        (?P<score>\d+(?:\.\d+)?) # 0.7909 など（使わないが吸収）
    \.(?P<ext>jpg|jpeg|png|webp|bmp)$
    """,
    re.IGNORECASE | re.VERBOSE
)

# 全角→半角（数字）用
_Z2H = str.maketrans("０１２３４５６７８９", "0123456789")

def normalize_gid(gid: str) -> str:
    """
    例:
      '画像1'  -> 'img1'
      '  画像１２  ' -> 'img12'
      '写真3'  -> 'img3'
      'img_4'  -> 'img4'
      'G-005'  -> 'g005' （任意）
    """
    s = (gid or "").strip()
    # 全角数字を半角に
    s = s.translate(_Z2H)
    # よくある日本語プレフィックスをimgへ
    s = re.sub(r"^(画像|寫真|写真)\s*", "img", s, flags=re.IGNORECASE)
    # もし 'img' が含まれていない & 先頭が英字でないなら、先頭に 'img' を補う（任意）
    if not re.match(r"^[A-Za-z]", s):
        s = "img" + s
    # 許容文字以外を除去（英数字とアンダースコア以外は消す）
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    # 小文字化
    s = s.lower()
    # 連続アンダースコア整理
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "img"

def parse_name(p: Path):
    m = PATTERN.match(p.name.replace(" ", ""))  # スペースを無視してマッチ
    if not m:
        return None
    d = m.groupdict()
    gid_raw = d["gid"]            # 画像1 など
    gid = normalize_gid(gid_raw)  # ← ここで img系に正規化
    phase = "before" if d["phase"] == "1" else "after"
    date  = d["date"]             # 230421
    idx   = d["idx"]              # 0
    ext   = "." + d["ext"].lower()
    return {
        "gid": gid,
        "phase": phase,
        "date": date,
        "idx": idx,
        "ext": ext
    }

def build_new_name(info: dict):
    # 例: img1_230421_0_before.jpg
    return f"{info['gid']}_{info['date']}_{info['idx']}_{info['phase']}{info['ext']}"

def ensure_dirs():
    OUT_BEFORE.mkdir(exist_ok=True, parents=True)
    OUT_AFTER.mkdir(exist_ok=True, parents=True)

def main():
    assert ROOT.exists(), f"Not found: {ROOT}"
    ensure_dirs()

    files = [p for p in ROOT.rglob("*")
             if p.is_file() and p.suffix.lower() in EXTS
             and p.parent not in (OUT_BEFORE, OUT_AFTER)]
    print(f"[INFO] Found {len(files)} files to consider under: {ROOT}")

    moved, skipped, unmatched = 0, 0, []

    for p in files:
        info = parse_name(p)
        if not info:
            unmatched.append(p)
            continue

        new_name = build_new_name(info)
        dst_dir = OUT_BEFORE if info["phase"] == "before" else OUT_AFTER
        dst = dst_dir / new_name

        if DRY_RUN:
            print(f"[DRY] {p.name}  ->  {dst.relative_to(ROOT)}")
            moved += 1
        else:
            if dst.exists():
                # 既に同名がある場合は連番付与
                stem, ext = dst.stem, dst.suffix
                k = 1
                while (dst_dir / f"{stem}__{k}{ext}").exists():
                    k += 1
                dst = dst_dir / f"{stem}__{k}{ext}"
            shutil.move(str(p), str(dst))
            print(f"[MOVE] {p.name}  ->  {dst.relative_to(ROOT)}")
            moved += 1

    # レポート
    print("\n=== REPORT ===")
    print(f"Moved (or to move in DRY): {moved}")
    print(f"Skipped (already inside before/after): {skipped}")
    print(f"Unmatched pattern: {len(unmatched)}")
    if unmatched:
        print("Examples of unmatched:")
        for up in unmatched[:10]:
            print("  -", up.name)

if __name__ == "__main__":
    main()
