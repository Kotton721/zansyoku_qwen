# audit_csv_pairs.py
from __future__ import annotations
import re
import argparse
import pandas as pd
from pathlib import Path

# 例: [画像1-1]_230421_1_num2_0.5875.jpg
#      └gid──┘ └p┘ └date┘ └idx┘  └num┘ └score┘
PAT = re.compile(
    r"""
    ^\[
        (?P<prefix>画像|写真|寫真)?(?P<gid>\d+)
        -
        (?P<phase>[12])          # 1=before, 2=after
    \]
    \s*_?
        (?P<date>\d{6,8})        # 230421 等
        _
        (?P<idx>\d+)
        _
        num(?P<num>\d+)
        _
        (?P<score>\d+(?:\.\d+)?)
    \.(?P<ext>jpg|jpeg|png|webp|bmp)$
    """,
    re.IGNORECASE | re.VERBOSE
)

def parse_oldname(name: str):
    s = (name or "").strip()
    if not s or s.lower() == "nan":
        return None
    # ファイル名中の空白揺れ対策
    s = s.replace(" ", "")
    m = PAT.match(s)
    if not m:
        return None
    d = m.groupdict()
    return {
        "gid": int(d["gid"]),
        "phase": "before" if d["phase"] == "1" else "after",
        "date": d["date"],  # 文字列のまま比較
        "idx": int(d["idx"]),
        "num": int(d["num"]),
        "ext": d["ext"].lower(),
        "raw": name,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="元CSV (indi_data.csv など)")
    ap.add_argument("--out", default="csv_pair_audit_report.csv", help="結果レポートCSVの出力先")
    ap.add_argument("--strict-date", action="store_true",
                   help="食前/食後で日付の一致も必須にする（デフォルト: 同一gidのみ必須）")
    args = ap.parse_args()

    # 文字コード推定: UTF-8-SIG → Shift_JIS の順で試す
    try:
        df = pd.read_csv(args.csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(args.csv, encoding="shift_jis")

    COL_B = "食前画像"
    COL_A = "食後画像"
    if COL_B not in df.columns or COL_A not in df.columns:
        raise SystemExit(f"CSVに『{COL_B}』『{COL_A}』列が見つかりません。")

    rows = []
    bad = 0
    for i, r in df.iterrows():
        b_raw = str(r.get(COL_B, "")).strip()
        a_raw = str(r.get(COL_A, "")).strip()
        b = parse_oldname(b_raw)
        a = parse_oldname(a_raw)

        status = "OK"
        reason = ""

        if b is None and a is None:
            status = "MISSING_BOTH"
            reason = "both empty or unparsable"
        elif b is None:
            status = "MISSING_BEFORE"
            reason = "before empty or unparsable"
        elif a is None:
            status = "MISSING_AFTER"
            reason = "after empty or unparsable"
        else:
            # フェーズ自体が逆/不正（例: 両方before/after）もチェック
            if b["phase"] != "before" or a["phase"] != "after":
                status = "PHASE_MISMATCH"
                reason = f"phases: before={b['phase']}, after={a['phase']}"

            # gid一致必須
            if status == "OK" and b["gid"] != a["gid"]:
                status = "GID_MISMATCH"
                reason = f"gid before={b['gid']} after={a['gid']}"

            # 厳密モードならdate一致も必須
            if status == "OK" and args.strict_date and (b["date"] != a["date"]):
                status = "DATE_MISMATCH"
                reason = f"date before={b['date']} after={a['date']}"

        if status != "OK":
            bad += 1

        rows.append({
            "row": i + 1,
            "before": b_raw,
            "after": a_raw,
            "b_gid": b["gid"] if b else "",
            "b_date": b["date"] if b else "",
            "b_idx":  b["idx"]  if b else "",
            "a_gid": a["gid"] if a else "",
            "a_date": a["date"] if a else "",
            "a_idx":  a["idx"]  if a else "",
            "status": status,
            "reason": reason,
        })

    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

    total = len(rows)
    print(f"[DONE] Report: {out}")
    print(f"[SUMMARY] total={total}, not_pair_or_problem={bad}, ok={total-bad}")
    print("Statuses used: OK, MISSING_BOTH, MISSING_BEFORE, MISSING_AFTER, PHASE_MISMATCH, GID_MISMATCH, DATE_MISMATCH")

if __name__ == "__main__":
    main()



