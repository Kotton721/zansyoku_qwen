# qwen_ab_pred (before + after)

食前画像と食後画像を比較し、食後に残っている食材ラベルを Qwen2.5-VL に出力させるワークフローです。ペア CSV を元にバッチ推論し、JSONL/CSV で残食予測を保存します。

## データ前提
- 画像は `dataset/indi_image/images/before` と `dataset/indi_image/images/after` のように分けて配置します。`convert2.py` / `convert_2_pair.py` で生データのファイル名を正規化し、before/after へ振り分ける補助が入っています（`DRY_RUN` を true にして挙動確認可能）。
- 推論入力はペア一覧 CSV です。サンプル `pairs_new.csv` は `before_new` / `after_new` カラムで絶対パスを持っています（デフォルト列名を変える場合は引数で指定）。
- ラベル集合は CVAT の XML（例: `qwen/qwen_after_only/datasets/indi_image/images/after_img_all_xml/annotations.xml`）から `<labels>` を読み込みます。`<attribute name="subclass">` が定義されていれば評価時にサブクラス指標も算出できます。

## 推論
単一モデルを回す最小例:
```bash
python3 pred_qwen_ab_ingred.py \
  --pairs_csv pairs_new.csv \
  --before-col before_new \
  --after-col after_new \
  --xml ../qwen_after_only/datasets/indi_image/images/after_img_all_xml/annotations.xml \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device_map auto --dtype auto --max_new_tokens 128 --temperature 0.0
```
- 出力はデフォルトで `results_remaining_pairs__{CSV名}__{モデル名}.jsonl` と `results_remaining_pairs__{CSV名}.csv` が同じフォルダに生成されます。`--out-jsonl` / `--out-csv` で変更可能。
- `run_qwen_ab.sh` は 3B/7B/32B の各モデルを同じペア CSV で一括実行するバッチ例です。冒頭のパスを環境に合わせて書き換えてください。

## 評価
`eval_ab.py` で leaf/subclass の PRF を計算できます。
```bash
python3 eval_ab.py \
  --pred ab_results/Qwen7B.jsonl \
  --gt_cvat_xml ../qwen_after_only/datasets/indi_image/images/after_img_all_xml/annotations.xml \
  --report_prefix reports/eval_qwen7b
```
- GT を CSV/JSONL で持っている場合は `--gt` で渡し、サブクラス定義 XML が別の場合は `--subclass_xml` で指定します。
- `--oov_policy penalize` を使うと語彙外ラベルを FN/FP として扱います（デフォルトは無視）。

## 補助スクリプトと成果物
- `convert2.py` / `convertfile.py` 系: 元画像ファイル名を解析して before/after に整列し、対応表 (`filename_map.csv/json`) を生成。
- `convert_2_pair.py` / `check_pair.py`: 正規化後のファイル名対応表からペア CSV を作る、存在確認を行うなどのデータ監査。
- `ab_results` / `result_Qwen*.json`: 実際に動かしたモデルの推論サンプル結果。フォーマット確認用に利用できます。
