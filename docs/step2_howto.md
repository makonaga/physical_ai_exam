# 段階2 実行手順：形・大きさがランダムな食材片（にんじん）

段階1の立方体を、にんじんを切った小片（ランダムな凸多面体）に置き換える。1 回に掴むのは 1 片。
手順の流れ（CPU でデータ生成 → GPU で学習 → CPU で評価）とスペースの切り替え方は
[段階1](step1_howto.md) と同じ。

## 全体の流れ

| 手順 | インスタンス | 内容 | 所要時間の目安 |
|---|---|---|---|
| A | `ml.c5.9xlarge` | 学習データの生成（成功 600 エピソード） | 15 分程度 |
| B | `ml.g5.xlarge` | ACT の学習（60,000 ステップ） | 約 80 分 |
| C | `ml.c5.9xlarge` | 評価（初めて見る形 50 回など 4 種類） | 1 種類あたり数分 |

インスタンスを起動したら、毎回まず次の 3 行を実行する。

```bash
cd ~/user-default-efs/physical_ai_exam
git pull
bash scripts/setup_sagemaker.sh
```

## A. 学習データの生成（ml.c5.9xlarge）

```bash
bash scripts/generate_dataset.sh pot_pieces_v1 600 --task pieces
```

終わったら `head -25 data/pot_pieces_v1/generation_stats.json` で結果を確認する
（`episodes` が 600、`expert_success_rate` が 0.95 以上なら次へ進んでよい）。

## B. ACT の学習（ml.g5.xlarge）

```bash
bash scripts/train_act.sh pot_pieces_v1 act_pot_pieces_v1 --steps=60000
```

データが段階1の 2 倍なので、学習ステップも 1.5 倍にしている。5 分後に
`tail -n 5 logs/train_act_pot_pieces_v1_*.log` で `loss` が下がっていることを確認する。

## C. 評価（ml.c5.9xlarge）

次の 4 つを、1 つずつ（前のログに `完了` が出てから）実行する。

```bash
# 1. 段階2のモデル × 初めて見る形（評価用の形状セット）… 本命
bash scripts/evaluate.sh act_pot_pieces_v1 50 --task pieces

# 2. 段階2のモデル × 学習で見た形（学習用の形状セット）… 1 との差で「初めて見る形への強さ」を見る
bash scripts/evaluate.sh act_pot_pieces_v1 50 --task pieces --piece-set carrot_train --name act_pot_pieces_v1_seen

# 3. 段階1のモデル（立方体だけで学習）× 初めて見る形 … 形を変えて学習した効果を見る
bash scripts/evaluate.sh act_pot_cube_v1 50 --task pieces --name act_pot_cube_v1_on_pieces

# 4. 段階2のモデル × 立方体 … 段階1の課題を解けなくなっていないか
bash scripts/evaluate.sh act_pot_pieces_v1 50 --task cube --name act_pot_pieces_v1_on_cube
```

結果はそれぞれ `cat outputs/eval/<名前>/summary.json` で確認する
（名前は `act_pot_pieces_v1`、`act_pot_pieces_v1_seen`、`act_pot_cube_v1_on_pieces`、`act_pot_pieces_v1_on_cube`）。
動画は `notebooks/view_eval_videos.ipynb` の `NAME` を変えて再生できる。

## 結果の共有

1. `head -25 data/pot_pieces_v1/generation_stats.json`（A）
2. 学習ログの最後の行（`grep "step:60K" logs/train_act_pot_pieces_v1_*.log | tail -n 1`）（B）
3. 4 つの評価の `summary.json` の `success_rate` と `outcomes`（C）

## 設計メモ

### 食材片の形（`src/so101_food/shapes/`）

- にんじん（半径 11〜16mm の円柱）を、ランダムな平面で切って凸多面体を作る。
  切り方は「乱切り」（正多面体の面の向きをランダムに回して揺らした 6〜12 枚の平面）と
  「角切り」（直交する 3 組の平面）の 2:1。
- 交差は半空間の交わりとして計算する（scipy の HalfspaceIntersection）。元の形も切断面も凸なので、結果も凸。
- 最小幅（一番薄い方向の厚み）15mm 以上、最大寸法 35mm 以下のものだけ残す（薄切りは対象外）。
- 机に置いて安定する向きに回し、外接箱の一番薄い軸を鉛直にそろえて書き出す。
- 学習用 300 個（`assets/pieces/carrot_train`）と評価用 100 個（`assets/pieces/carrot_test`）を別の seed で作り、
  リポジトリに含める。高さ 16〜29mm、上から見た寸法 17〜35mm、質量は体積 × 0.95g/cm³。
- 色はにんじんの橙色を基本に、エピソードごとに明るさ・色味を ±8% 揺らす。

### エキスパートの改良

- **掴み方**: 小片の頂点を上から見た凸多角形にし、幅が最も狭い方向（回転キャリパー法）で挟む。
  掴む位置はその方向と直交方向それぞれの中央、高さは小片の上下の中央（指先が床に当たらないよう下限 10mm）。
- **握る強さ**: 以前は「完全に閉じる角度」を目標にしていたが、目標と実際の差が握る力になるため強すぎ、
  小片が指先の小さな当たり判定に 5〜10mm 食い込んで、指を開いても離れないことがあった。
  閉じる目標を「パッド間が小片の幅より 10mm 狭い角度」に変えて解決した
  （学習用の形で成功率 95.0% → 99.5%）。
- **離すとき**: 指を大きく開く（70°）。

この 2 つの改良で、エキスパートの成功率は次のとおり（各 100〜200 回、開発環境）。

| 対象 | 成功率 |
|---|---|
| 学習用の形（carrot_train） | 99.5% |
| 評価用の形（carrot_test） | 99.0% |
| 立方体（段階1と同じ条件） | 100% |

段階1のデータはこの改良の前のエキスパートで作った（段階1の結果はそのままで比較できる）。

### シミュレーションの注意点

- 1 つの環境に 20 個の食材片を入れておき、エピソードごとに 1 個を置く（データ生成では 10 試行ごとに
  形状セットから選び直す）。評価ではエピソード番号から決まる 1 個を使う（並列数によらず同じ条件）。
- so101-nexus は使わない片を同じ場所に重ねて置き、reset の途中で一時的に接触を有効にする。片が多いと
  重なった片どうしの接触が数百件になり MuJoCo がメモリ不足で止まるため、reset の前に待機中の片を離して並べる。
