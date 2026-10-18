# 段階2 実行手順：散らばった食材片から一番近い片を鍋に入れる

机の上に、形・大きさがランダムなにんじんの小片を **5〜20 個**（エピソードごとにランダム）散らばせ、
その中から **ロボットに一番近い片** を 1 つ掴んで鍋に入れる。1 個入れたらエピソードは終わり。

段階1との違い：

| 項目 | 段階1 | 段階2 |
|---|---|---|
| 机の上の物体 | 25mm 角の立方体 1 個 | にんじんの小片 5〜20 個（形・大きさ・向き・色味がランダム） |
| 掴む物体 | その 1 個 | ロボットの根元に一番近い片 |
| 開始姿勢 | so101-nexus の休止姿勢 | 手先を上げた姿勢（下の「設計メモ」参照） |
| データ名 | `pot_cube_v1` | `pot_multi_v1`（成功 1,000 エピソード） |
| 学習名 | `act_pot_cube_v1` | `act_pot_multi_v1`（80,000 ステップ） |

## 全体の流れ

| 手順 | インスタンス | 内容 | 所要時間の目安 |
|---|---|---|---|
| 0 | `ml.c5.9xlarge` | 古いデータの削除 | 1 分 |
| A | `ml.c5.9xlarge` | 学習データの生成（成功 1,000 エピソード）と動画での確認 | 30 分程度 |
| B | `ml.g5.xlarge` | ACT の学習（80,000 ステップ） | 約 1 時間 45 分 |
| C | `ml.c5.9xlarge` | 評価（初めて見る形・学習で見た形、各 100 エピソード） | 1 種類あたり 5 分程度 |

インスタンスの切り替え方は [段階1](step1_howto.md) と同じ（スペースを Stop → Instance を変更 → Run space →
Open JupyterLab）。起動したら毎回、ターミナル（File → New → Terminal）で次の 3 行を実行する。

```bash
cd ~/user-default-efs/physical_ai_exam
git pull
bash scripts/setup_sagemaker.sh
```

`git pull` が `Your local changes ... would be overwritten` で止まったときは、ノートブックのタブを閉じてから
`git checkout -- notebooks/` を実行し、もう一度 `git pull` する（ノートブックの実行結果が残っているだけなので消してよい）。

以下のコマンドは、すべて `~/user-default-efs/physical_ai_exam` で実行する。

## 0. 古いデータの削除（ml.c5.9xlarge）

以前に作った「机の上に片が 1 個だけ」のデータ（`pot_pieces_v1`）は使わないので消す。
食材片の形状セットも作り直した（置いた直後に転がる形を除いた）ため、このデータとは形が合わない。

```bash
rm -rf data/pot_pieces_v1 outputs/dataset_videos/pot_pieces_v1
```

何も表示されなければ完了。

## A. 学習データの生成（ml.c5.9xlarge）

### A-1. 生成を始める

```bash
bash scripts/generate_dataset.sh pot_multi_v1 1000 --task multi
```

すぐに次のように表示され、生成は裏で進む（ブラウザやターミナルを閉じても止まらない。スペースを Stop すると止まる）。

```
裏で実行を始めました（PID 12345）。
  進み具合を見る : tail -f .../logs/generate_pot_multi_v1_20261018_101500.log    （見るのをやめるときは Ctrl+C。処理は止まりません）
  処理を止める   : kill 12345
```

### A-2. 進み具合を見る

```bash
tail -f logs/generate_pot_multi_v1_*.log
```

- `[worker 05] 12/28 (試行 12, 410s)` のような行が流れる（ワーカー 5 番が 28 本中 12 本目を保存した、という意味）。
- `[mp4 @ ...] Starting second pass: moving the moov atom ...` という行も大量に出るが、動画を書き出すときの
  お知らせでエラーではない。
- 見るのをやめるときは Ctrl+C（生成は止まらない）。

### A-3. 終わったか確認する（20〜25 分後）

ログの最後に `完了: .../data/pot_multi_v1` と出ていれば終わり。結果を確認する。

```bash
head -30 data/pot_multi_v1/generation_stats.json
```

- `"episodes": 1000` … 学習に使う成功エピソード数
- `"expert_success_rate"` … エキスパートの成功率。0.95 以上なら問題なし（開発環境では 0.99 以上）
- `"expert_outcomes"` … 試行の内訳。`success` 以外（失敗）は学習データに入っていない

### A-4. 生成したデータを動画で見る

```bash
bash scripts/export_dataset_videos.sh pot_multi_v1 6
```

1〜3 分で終わり、最後に `完了: .../outputs/dataset_videos/pot_multi_v1` と出る。できる動画は次の 7 本。

- `ep000_13pieces.mp4` など 6 本 … 1 エピソードずつ（左: 俯瞰カメラ、右: 手首カメラ。左上にエピソード番号・
  机の上の片の数・経過時間）
- `grid_9.mp4` … 9 エピソードの俯瞰カメラを 3×3 に並べた一覧

見方：

1. 左のファイル一覧で `user-default-efs` → `physical_ai_exam` → `notebooks` → `view_videos.ipynb` をダブルクリック
2. カーネルを聞かれたら「Python 3.12 (pai)」を選ぶ
3. 上のメニューの Run → Run All Cells
4. 下の方に動画が 7 本並ぶので ▶ で再生

俯瞰カメラでは画面の左上にロボットの根元がある。**左上に一番近い片** を掴んで右の鍋に入れ、
開始姿勢に戻っていれば正しい。

## B. ACT の学習（ml.g5.xlarge）

インスタンスを `ml.g5.xlarge` に切り替え、最初の 3 行（`cd` / `git pull` / `setup_sagemaker.sh`）を実行する。
`setup_sagemaker.sh` の最後に `CUDA 利用可能: True（NVIDIA A10G）` と出れば GPU が使える。

```bash
bash scripts/train_act.sh pot_multi_v1 act_pot_multi_v1 --steps=80000
```

- データが段階1の約 3 倍なので、学習ステップも 2 倍（80,000）にしている。約 1 時間 45 分かかる。
- 5 分ほどたったら進み具合を確認する。

  ```bash
  tail -n 5 logs/train_act_pot_multi_v1_*.log
  ```

  `step:4K ... loss:0.xxx ...` のような行が出ていれば正常（step が増え、loss が下がっていく）。
- 終わったかの確認：

  ```bash
  grep "step:80K" logs/train_act_pot_multi_v1_*.log | tail -n 1
  ls outputs/train/act_pot_multi_v1/checkpoints/
  ```

  1 行目に 80K ステップ時点の loss が出て、2 行目に `080000` と `last` があれば完了。
- 学習中はスペースを Stop しないこと（Stop すると学習も止まる）。

## C. 評価（ml.c5.9xlarge）

インスタンスを `ml.c5.9xlarge` に戻し、最初の 3 行を実行してから、次の 2 つを 1 つずつ実行する
（前のログに `完了` が出てから次を始める）。

```bash
# 1. 初めて見る形（評価用の形状セット）… 本命
bash scripts/evaluate.sh act_pot_multi_v1 100 --task multi

# 2. 学習で見た形（学習用の形状セット）… 1 との差で「初めて見る形への強さ」を見る
bash scripts/evaluate.sh act_pot_multi_v1 100 --task multi --piece-set carrot_train --name act_pot_multi_v1_seen
```

それぞれ 5 分程度。進み具合は `tail -f logs/eval_act_pot_multi_v1_*.log`（2 つ目は `logs/eval_act_pot_multi_v1_*.log`
の新しい方。`ls -t logs/ | head -3` で新しい順に確認できる）。

結果の確認：

```bash
cat outputs/eval/act_pot_multi_v1/summary.json
cat outputs/eval/act_pot_multi_v1_seen/summary.json
```

- `success_rate` … 成功率（一番近い片を鍋に入れられた割合）
- `outcomes` … 内訳。`success` 成功 / `wrong_piece` 一番近くない片を持ち上げた・鍋に入れた /
  `not_grasped` 一番近い片を持ち上げられなかった / `missed_pot` 持ち上げたが鍋に入らなかった
- `by_n_pieces` … 机の上の片の数（5〜9 個、10〜14 個、15〜19 個、20 個）ごとの成功率

動画は先頭 6 エピソード分が `outputs/eval/<名前>/videos/` に保存される。`view_videos.ipynb` の 1 行目を
`FOLDER = "outputs/eval/act_pot_multi_v1/videos"` に変えて Run All Cells で再生できる。

## 結果の共有

次の 3 つを貼ってもらえれば、`docs/step2_results.md` にまとめる。

1. `head -30 data/pot_multi_v1/generation_stats.json`（A）
2. `grep "step:80K" logs/train_act_pot_multi_v1_*.log | tail -n 1`（B）
3. 2 つの評価の `summary.json`（C）

## 設計メモ

### 食材片の形（`src/so101_food/shapes/`）

- にんじん（半径 11〜16mm の円柱）を、ランダムな平面で切って凸多面体を作る。
  切り方は「乱切り」（正多面体の面の向きをランダムに回して揺らした 6〜12 枚の平面）と
  「角切り」（直交する 3 組の平面）の 2:1。
- 交差は半空間の交わりとして計算する（scipy の HalfspaceIntersection）。元の形も切断面も凸なので、結果も凸。
- 最小幅（一番薄い方向の厚み）15mm 以上、最大寸法 35mm 以下のものだけ残す（薄切りは対象外）。
- 机に置いて安定する向きに回し、外接箱の一番薄い軸を鉛直にそろえて書き出す。
  さらに、**重心を真下に下ろした点が接地面の端から 3.5mm 以上内側にある向き** だけを使う。
  以前は、調べた 300 個のうち 39 個が、置いた直後にシミュレーションの中で転がったり回ったりしていた
  （重心が接地面の端から 2mm 程度しか離れていない片）。この条件を加えて作り直し、400 個すべてが動かなくなった。
- 学習用 300 個（`assets/pieces/carrot_train`）と評価用 100 個（`assets/pieces/carrot_test`）を別の seed で作り、
  リポジトリに含める。高さ 15〜29mm、上から見た寸法 17〜35mm、質量は体積 × 0.95g/cm³。
- 色はにんじんの橙色を基本に、片ごとに明るさ・色味を ±8% 揺らす。

### 片の並べ方（`src/so101_food/env/pot_task.py` の `ClutterSpec`）

- 片の数は 5〜20 個から一様に選ぶ（先に数を決めてから位置を探すので、数の分布は偏らない）。
- 置く範囲は x 11〜34cm、y −8.5〜13cm（ロボットの根元が原点）。「俯瞰カメラに写る」「鍋と 1cm 以上離れる」
  「根元から 12cm 以上離れる（真上から掴めるのは根元から 12〜30cm）」で決めた。
- 片どうしは、上から見た形（凸多角形）が 3mm 以上離れるように置く（重ならない。くっつきそうな配置もある）。
- **掴む片 = ロボットの根元に一番近い片**（上から見た形の重心までの距離で比べる）。
  2 番目に近い片との差が 1cm 未満の配置は、どれが一番近いか画像で見分けにくいので置き直す。
- 下の「掴み方」で、エキスパートが他の片に指を当てずに掴めない配置も置き直す（ほとんど起きない）。
  評価でも同じ条件の配置を使う。
- 掴む片までの距離は、ほとんどが根元から 13〜16cm（中央値 14.5cm）。片が少ないときは 24cm 程度のこともある。

### 開始姿勢

so101-nexus の休止姿勢（肘 90°、手首 37.8°）では、手先が机から約 6cm の高さで作業範囲の上にかぶさり、
俯瞰カメラから根元に近い片（＝掴む片）が隠れてしまう。そこで開始姿勢を
（肩パン 0°, 肩リフト −70°, 肘 35°, 手首 90°, 手首ロール 0°, グリッパ −10°）に変えた。手先は机から約 14cm で、
俯瞰カメラから作業範囲がまったく隠れず、手首カメラにも作業範囲の約 9 割が写る。エピソードの最後もこの姿勢に戻る。

### エキスパートの改良（`src/so101_food/expert/pot_expert.py`）

- **掴む向き**: 上から見て一番細い方向に加え、0〜180° を 10° ごとに試し、幅の狭い順に候補にする。
  各候補について、今の姿勢 → 上空 → 把持姿勢と動く間に、手先（指・手首カメラの台座）が
  他の片から 4mm 以上、自分の腕（肩・上腕・前腕）からも 4mm 以上離れているかを、関節角度の補間経路上の数点で調べ、
  最初に条件を満たしたものを使う。
  - 自分の腕も調べるのは、根元に近い片を手首を大きくひねって掴もうとすると、手首カメラの台座が肩に当たって
    下りられないことがあったため。
- **開く幅**: 他の片があるときは「片の幅 + 両側 10mm」だけ開く（最大 35°）。大きく開くより隣の片に当たりにくい。
- 握る強さ（片の幅より 10mm 狭い目標まで閉じる）、離すときに大きく開く（70°）は以前と同じ。

エキスパートの成功率（開発環境）：

| 対象 | 成功率 |
|---|---|
| 学習用の形（carrot_train）、5〜20 個 | 200 / 200 |
| 評価用の形（carrot_test）、5〜20 個 | 99 / 100 |

### 結果の分類

| 分類 | 意味 |
|---|---|
| `success` | 一番近い片が鍋に入った |
| `wrong_piece` | 一番近い片は持ち上げず、別の片を鍋に入れた・2cm 以上持ち上げた |
| `not_grasped` | 一番近い片を持ち上げられなかった |
| `missed_pot` | 一番近い片を持ち上げたが鍋に入らなかった |
