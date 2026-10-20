# SO-101 食材投入タスク（Physical AI 最終課題）

切った食材の小片を SO-101 アームで掴んで鍋に入れるタスクを、MuJoCo シミュレータ上で
スクリプテッド・エキスパートが自動生成したデータから、LeRobot の模倣学習（ACT と VLA の SmolVLA）で獲得する。

- ロボット: SO-101（[so101-nexus](https://github.com/johnsutor/so101-nexus) の MuJoCo モデル・環境を土台に使用）
- 学習: [LeRobot](https://github.com/huggingface/lerobot)（ACT と、VLA の SmolVLA）
- 実行環境: Amazon SageMaker Studio（JupyterLab）。データ生成・評価は CPU インスタンス（`ml.c5.9xlarge`）、
  学習は GPU インスタンス（ACT は `ml.g5.xlarge`、SmolVLA は `ml.g5.2xlarge`）

## 進捗

| 段階 | 内容 | 状態 |
|---|---|---|
| 0 | 環境構築、グリッパ寸法・作業領域の確認、把持できるかの確認 | 完了。結果: [docs/step0_notes.md](docs/step0_notes.md) |
| 1 | 立方体 1 個でエキスパート → データ生成 → ACT 学習 → 評価 | 完了（成功率 82%）。手順: [docs/step1_howto.md](docs/step1_howto.md)、結果: [docs/step1_results.md](docs/step1_results.md) |
| 2 | 形・大きさがランダムなにんじんの小片を 5〜20 個散らばせ、ロボットに一番近い片を鍋に入れる | 学習・評価まで完了（初めて見る形で成功率 56%。掴む片が遠いと掴み損ねる）。原因を調査中。手順: [docs/step2_howto.md](docs/step2_howto.md)、結果: [docs/step2_results.md](docs/step2_results.md) |
| 3 | にんじん・じゃがいも・大根の片を混ぜて 7〜26 個散らばせ、指示文で指定した食材の一番近い片を鍋に入れる（SmolVLA と、指示文を読まない ACT を比較） | 食材らしい形（じゃがいも = 丸みのある乱切り、大根 = いちょう切り）と食材ごとの重さ、片の数 7〜26 個（段階2の 3 割増し）に変更し、学習データを作り直した（1,500 エピソード）。学習・評価はこれから。手順: [docs/step3_howto.md](docs/step3_howto.md)、結果: [docs/step3_results.md](docs/step3_results.md) |
| 4 | 動画・レポート・リポジトリ整理 | 未着手 |
| ― | ACT を VLA（SmolVLA）に置き換える（段階2のデータで学習し、ACT と比較） | 学習・評価まで完了（初めて見る形で成功率 40%。ACT は 56%。推論し直す間隔を短くしても改善せず 28%）。結果: [docs/vla_results.md](docs/vla_results.md)。手順: [docs/vla_howto.md](docs/vla_howto.md) |

## 次回の作業

段階3の学習・評価（実験 E1〜E6）。手順と順番は [docs/step3_howto.md](docs/step3_howto.md) の「全体の流れ」と B〜D。
1 日目は `ml.g5.2xlarge` で、E4（画像部分も学習する SmolVLA）の試運転のあと、E2（SmolVLA）を夜通し学習する。

```bash
cd ~/user-default-efs/physical_ai_exam
git pull
bash scripts/setup_sagemaker.sh
TRAIN_VLM=1 bash scripts/train_smolvla.sh pot_lang_v1 smolvla_vlm_trial --steps=200 --save_freq=200 --log_freq=20
# ↑ が終わってから（20〜30 分）
bash scripts/train_smolvla.sh pot_lang_v1 smolvla_pot_lang_v1
```

ACT の改善（掴む片が遠いと掴み損ねる原因の調査）は保留中。手順は [docs/step2_results.md](docs/step2_results.md) の 7。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/plan.md](docs/plan.md) | 当初の計画メモ（課題の要件・方針。段階2以降の細部は実装と異なる） |
| [docs/step0_notes.md](docs/step0_notes.md) | 段階0: 環境・グリッパ寸法・作業領域の計測結果 |
| [docs/step1_howto.md](docs/step1_howto.md) | 段階1: SageMaker での実行手順（インスタンスの切り替え方を含む） |
| [docs/step1_results.md](docs/step1_results.md) | 段階1: 結果（データ生成・学習・評価） |
| [docs/step2_howto.md](docs/step2_howto.md) | 段階2: 実行手順と設計メモ（食材片の形・並べ方・開始姿勢・エキスパートの掴み方） |
| [docs/step2_results.md](docs/step2_results.md) | 段階2: 結果（学習・評価）と考察 |
| [docs/vla_howto.md](docs/vla_howto.md) | VLA（SmolVLA）: 選んだ理由、学習・評価の手順 |
| [docs/vla_results.md](docs/vla_results.md) | VLA（SmolVLA）: 結果（学習・評価）と ACT との比較 |
| [docs/step3_howto.md](docs/step3_howto.md) | 段階3: 実行手順と設計メモ（じゃがいも・大根の形、指示文、対象の選び方） |
| [docs/step3_results.md](docs/step3_results.md) | 段階3: 結果（データ生成・学習・評価） |

## セットアップ（SageMaker Studio）

1. JupyterLab スペースを作成する（イメージ: SageMaker Distribution 最新、ストレージ 50GB 程度）。
2. ターミナル（File → New → Terminal）でリポジトリを取得し、セットアップスクリプトを実行する。

   ```bash
   cd ~/user-default-efs
   git clone https://github.com/makonaga/physical_ai_exam.git
   cd physical_ai_exam
   bash scripts/setup_sagemaker.sh
   ```

   - 仮想環境は `~/.venvs/pai`（ホーム配下なのでスペースを再起動しても残る）に作る。
   - OS パッケージ（EGL / OSMesa / ffmpeg）は再起動で消えるため、
     **スペースを起動するたびにスクリプトを再実行する**（2 回目以降は数十秒で終わる）。
   - GPU の有無とドライバのバージョンから PyTorch の CUDA 版を自動で選ぶ。
     手動で指定する場合は `TORCH_BACKEND=cu126 bash scripts/setup_sagemaker.sh` のようにする。
3. ノートブックを開き、カーネル **「Python 3.12 (pai)」** を選んで実行する。

`git pull` が `Your local changes ... would be overwritten` で止まったときは、ノートブックのタブを閉じてから
`git checkout -- notebooks/` を実行し、もう一度 `git pull` する（ノートブックの実行結果が残っているだけ）。

## タスクの種類

データ生成・評価のスクリプトに `--task` で指定する。

| `--task` | 内容 | 使った段階 |
|---|---|---|
| `cube` | 25mm 角の立方体 1 個を鍋に入れる | 段階1 |
| `pieces` | 形・大きさがランダムな食材片 1 個を鍋に入れる | （比較用） |
| `multi` | 食材片を 5〜20 個散らばせ、ロボットに一番近い片を鍋に入れる | 段階2 |
| `lang` | にんじん・じゃがいも・大根の片を混ぜて 7〜26 個散らばせ、指示文で指定した食材の一番近い片を鍋に入れる | 段階3 |

どのタスクも、ポリシーの入力は俯瞰カメラ・手首カメラの画像（224×224）と 6 関節の角度、
出力は 6 関節の目標角度（25Hz）。物体の真の位置はポリシーに与えない。
指示文（例: `Put the potato piece closest to the robot into the pot.`）はデータに記録され、SmolVLA は入力に使う
（ACT は使わない）。

## スクリプト

時間のかかる処理は、ターミナルから裏で実行するスクリプトにしている。
ログは `logs/<処理>_<名前>_<日時>.log`（例: `logs/generate_pot_multi_v1_20261018_101500.log`）に書かれ、
`tail -f logs/<ファイル名>` で進み具合を見られる。新しい順の一覧は `ls -t logs/ | head -3`。

| スクリプト | 内容 | 向いているインスタンス |
|---|---|---|
| `scripts/setup_sagemaker.sh` | 環境構築（スペース起動のたびに実行） | どれでも |
| `scripts/generate_dataset.sh <名前> <エピソード数> [--task ...]` | エキスパートで学習データを生成（並列） | CPU（vCPU が多いもの） |
| `scripts/export_dataset_videos.sh <名前> [本数]` | 生成したデータを再生用の動画に書き出す | どれでも |
| `scripts/train_act.sh <データ名> <実行名> [--steps=...]` | ACT の学習 | GPU |
| `scripts/train_smolvla.sh <データ名> <実行名> [--steps=...]` | SmolVLA（VLA）の追加学習。先頭に `TRAIN_VLM=1` を付けると画像を読む部分と言語モデルも学習する | GPU |
| `scripts/evaluate.sh <実行名> <エピソード数> [--task ...]` | 学習済みポリシーの評価（並列）。`--noise`・`--temporal-ensemble` で推論のしかたを変えられる | CPU（vCPU が多いもの） |
| `scripts/analyze_eval.sh <評価名...> [--dataset <データ名>]` | 評価結果を距離・片の数ごとに比較する | どれでも |
| `scripts/demo_language.sh <実行名> [配置の数]` | 同じ配置で指示文の食材だけを変えてポリシーを動かすデモ動画を作る（段階3） | CPU |
| `scripts/sync_progress_s3.sh <実行名> <S3 のコピー先>` | 学習の進み具合を 5 分ごとに S3 にコピーする（手元のパソコンから AWS CLI で確認するため） | 学習中のインスタンス |

## ノートブック

| ノートブック | 内容 |
|---|---|
| `notebooks/00_env_check.ipynb` | GPU・描画バックエンドの確認、ランダム動作の動画、グリッパ開口幅、真上把持の作業領域、小片の把持テスト |
| `notebooks/view_videos.ipynb` | 生成データ・評価の動画をノートブック上で再生する（1 行目の `FOLDER` を見たいフォルダにして Run All Cells） |
| `notebooks/watch_training.ipynb` | 裏で動いている学習の進み具合を 1 分ごとに表示し、終わるまで待つ（1 行目の `RUN` を学習の実行名にして Run All Cells） |

## ディレクトリ構成

```
├── src/so101_food/
│   ├── utils/     # 描画バックエンド選択、描画品質、動画保存、並列実行
│   ├── sim/       # グリッパ開口幅の計測
│   ├── expert/    # 真上把持の IK、掴み方の選択、エキスパート（把持 → 運搬 → 投入 → 帰還）
│   ├── shapes/    # 食材片の形状生成器（乱切り・角切り・丸みのある乱切り・いちょう切りの凸多面体）
│   ├── env/       # 鍋投入タスクの環境（食材片の並べ方を含む）、カメラ（俯瞰・手首）の描画
│   ├── datagen/   # 成功フィルタ付きデータ生成（並列、LeRobot v3 形式）、動画への書き出し
│   ├── vla/       # VLA の追加学習の補助（カメラ名の対応付け）
│   └── eval/      # 学習済みポリシーの評価（並列）、評価結果の分析
├── assets/pieces/ # 事前生成した食材片の形状セット（にんじん・じゃがいも・大根 × 学習用 300 個・評価用 100 個）
├── notebooks/     # SageMaker で実行するノートブック
├── scripts/       # セットアップ・データ生成・学習・評価のスクリプト
├── tests/         # 動作確認テスト（pytest）
└── docs/          # 計画メモ・手順書・結果（レポート素材）
```

次のフォルダは実行時に作られ、Git には含めない（`.gitignore`）。

| フォルダ | 中身 |
|---|---|
| `data/` | 生成した学習データ（LeRobot v3 形式。例: `data/pot_multi_v1/`） |
| `outputs/` | 学習済みモデル（`outputs/train/`）、評価結果（`outputs/eval/`）、データの動画（`outputs/dataset_videos/`）、デモ動画（`outputs/demo/`） |
| `logs/` | スクリプトのログ |

## テスト

```bash
source ~/.venvs/pai/bin/activate
python -m pytest -q
```

## 利用しているソフトウェア

- [so101-nexus](https://github.com/johnsutor/so101-nexus)（Apache-2.0）: SO-101 の MJCF、MuJoCo 環境
- [LeRobot](https://github.com/huggingface/lerobot)（Apache-2.0）: データセット形式、ACT の学習
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base)（Apache-2.0）: 事前学習済みの VLA。土台は [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)（Apache-2.0）
- [MuJoCo](https://github.com/google-deepmind/mujoco)（Apache-2.0）
