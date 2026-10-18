# SO-101 食材投入タスク（Physical AI 最終課題）

切った食材の小片を SO-101 アームで掴んで鍋に入れるタスクを、MuJoCo シミュレータ上で
スクリプテッド・エキスパートが自動生成したデータから、LeRobot の模倣学習（ACT）で獲得する。

- ロボット: SO-101（[so101-nexus](https://github.com/johnsutor/so101-nexus) の MuJoCo モデル・環境を土台に使用）
- 学習: [LeRobot](https://github.com/huggingface/lerobot)（ACT）
- 実行環境: Amazon SageMaker Studio（JupyterLab）

## 進捗

| 段階 | 内容 | 状態 |
|---|---|---|
| 0 | 環境構築、グリッパ寸法・作業領域の確認、把持できるかの確認 | 完了（SageMaker で確認済み）。結果: [docs/step0_notes.md](docs/step0_notes.md) |
| 1 | 直方体 1 個でエキスパート → データ生成 → ACT 学習 → 評価 | 完了（成功率 82%）。手順: [docs/step1_howto.md](docs/step1_howto.md)、結果: [docs/step1_results.md](docs/step1_results.md) |
| 2 | ランダム凸多面体（切断した食材片）への置き換え | 未着手 |
| 3 | 複数個・複数食材、分布外評価、薄切りの失敗デモ | 未着手 |
| 4 | 動画・レポート・リポジトリ整理 | 未着手 |

## セットアップ（SageMaker Studio）

1. JupyterLab スペースを作成する（イメージ: SageMaker Distribution 最新、ストレージ 50GB 程度。
   学習時は GPU インスタンス、例: `ml.g5.xlarge`）。
2. ターミナルでリポジトリを取得し、セットアップスクリプトを実行する。

   ```bash
   cd ~
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

## ノートブック

| ノートブック | 内容 |
|---|---|
| `notebooks/00_env_check.ipynb` | GPU・描画バックエンドの確認、ランダム動作の動画、グリッパ開口幅、真上把持の作業領域、小片の把持テスト |

出力（動画・図・計測結果）は `outputs/` に保存する（Git には含めない）。

## スクリプト

時間のかかる処理は、ターミナルから裏で実行するスクリプトにしている（ログは `logs/`）。
詳しい使い方は [docs/step1_howto.md](docs/step1_howto.md)。

| スクリプト | 内容 | 向いているインスタンス |
|---|---|---|
| `scripts/setup_sagemaker.sh` | 環境構築（スペース起動のたびに実行） | どれでも |
| `scripts/generate_dataset.sh` | エキスパートで学習データを生成（並列） | CPU（vCPU が多いもの） |
| `scripts/train_act.sh` | ACT の学習 | GPU |
| `scripts/evaluate.sh` | 学習済みポリシーの評価（並列） | CPU（vCPU が多いもの） |

## ディレクトリ構成

```
├── src/so101_food/
│   ├── utils/     # 描画バックエンド選択、描画品質、動画保存、並列実行
│   ├── sim/       # グリッパ開口幅の計測
│   ├── expert/    # 真上把持の IK、エキスパート（把持 → 運搬 → 投入 → 帰還）
│   ├── shapes/    # 食材小片の形状生成器（段階2）
│   ├── env/       # 鍋投入タスクの環境、カメラ（俯瞰・手首）の描画
│   ├── datagen/   # 成功フィルタ付きデータ生成（並列、LeRobot v3 形式）
│   └── eval/      # 学習済みポリシーの評価（並列）
├── notebooks/     # SageMaker で実行するノートブック
├── scripts/       # セットアップ・データ生成・学習・評価のスクリプト
├── tests/         # 動作確認テスト（pytest）
└── docs/          # 計測結果・設計メモ（レポート素材）
```

## テスト

```bash
source ~/.venvs/pai/bin/activate
python -m pytest -q
```

## 利用しているソフトウェア

- [so101-nexus](https://github.com/johnsutor/so101-nexus)（Apache-2.0）: SO-101 の MJCF、MuJoCo 環境
- [LeRobot](https://github.com/huggingface/lerobot)（Apache-2.0）: データセット形式、ACT の学習
- [MuJoCo](https://github.com/google-deepmind/mujoco)（Apache-2.0）
