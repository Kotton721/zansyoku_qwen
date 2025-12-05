# qwen_after_only (after image only)

食後画像だけを入力し、皿に残っている食材を Qwen2.5-VL で列挙するワークフローです。CVAT のラベル定義から語彙を取得し、予測を JSONL で保存します。

## データ前提
- 画像は `datasets/indi_image/images/after` など任意のディレクトリに配置します。`convertfile.py` で生データのファイル名を正規化し、before/after に振り分ける補助が入っています（`DRY_RUN=True` で試走可）。
- ラベル集合は `datasets/indi_image/images/after_img_all_xml/annotations.xml` のような CVAT XML を使用します。`<attribute name="subclass">` に紐づく値はサブクラス評価や集計で利用されます。

## 推論
`pred_qwen_ingred_detail.py` が単画像推論の入口です。XML から画像リストを取るか、ディレクトリを再帰で走査するかを選びます。
```bash
python3 pred_qwen_ingred_detail.py \
  --xml datasets/indi_image/images/after_img_all_xml/annotations.xml \
  --images-root datasets/indi_image/images/after \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device_map auto --dtype auto --max_new_tokens 128 --temperature 0.0
# あるいはディレクトリ指定のみ
# python3 pred_qwen_ingred_detail.py --after_dir datasets/indi_image/images/after --model ...
```
- フィルタ用オプション: `--pattern` でファイル名部分一致、`--range` で番号範囲（例: `10-200`）、`--include/--exclude` で特定ラベルを持つ画像を抽出。
- 出力ファイルは指定がなければ `results_remaining_{images-root名}_{モデル名}.jsonl` が作成されます。
- サブクラス単位での推論を直接行いたい場合は `vlm_remaining_by_subclass.py` を利用できます（細粒度ラベルをサブクラスへ射影し JSONL で保存）。

## 評価と集計
`eval.py` で leaf/subclass の指標を計算します。引数は `qwen_ab_pred` 版と同じで、GT を XML または CSV/JSONL から与えます。
```bash
python3 eval.py \
  --pred results_remaining_after_Qwen2.5-VL-7B-Instruct.jsonl \
  --gt_cvat_xml datasets/indi_image/images/after_img_all_xml/annotations.xml \
  --report_prefix reports/eval_after_only
```
- `make_perfile_summary_combined.py` は推論結果と GT を突き合わせてファイル別サマリを作る補助スクリプトです。
- `count_labels_subclasses.py` は XML 内のラベル・サブクラス分布を確認する簡易統計用です。

## 補助スクリプト
- `convertfile.py`: 元画像ファイル名を解析し before/after へ移動またはコピーしつつ、新しいファイル名へ正規化。
- `vlm_remaining_by_subclass.py`: 細粒度ラベルをサブクラスへマッピングしながら推論する変種。
- `quantilized_result/` や `results/` 配下に推論サンプルがあるのでフォーマット確認に利用できます。
