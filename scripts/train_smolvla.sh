#!/usr/bin/env bash
# SmolVLA（事前学習済みの VLA）を学習データで追加学習する（GPU インスタンス向け）.
#
#   bash scripts/train_smolvla.sh [データセット名] [実行名] [lerobot-train の追加オプション...]
#   例: bash scripts/train_smolvla.sh pot_multi_v1 smolvla_pot_multi_v1
#       bash scripts/train_smolvla.sh pot_multi_v1 smolvla_trial --steps=200 --save_freq=200 --log_freq=20  # 試運転
#
# 事前学習済みモデル lerobot/smolvla_base（約 4.5 億パラメータ。SO-100/101 のデータで事前学習済み）から始める。
# 初回は Hugging Face Hub からこのモデルと、土台の言語・画像モデル SmolVLM2-500M をダウンロードする
# （合わせて数 GB。~/.cache/huggingface に保存され、2 回目からは使い回す）。
# 学習データのカメラ名（overhead / wrist）は、事前学習済みモデルのカメラ名に自動で対応付ける
# （src/so101_food/vla/rename_map.py）。
#
# GPU が複数あるインスタンス（例: ml.g5.12xlarge は A10G × 4）では、全部の GPU で並列に学習する
# （accelerate で GPU ごとに 1 プロセス）。1 GPU あたりのバッチを 64 ÷ GPU 数にして、合計のバッチ（64）と
# ステップ数は 1 GPU のときと同じにするので、同じ学習をより短い時間で行える。
#
# 出力: outputs/train/<実行名>/checkpoints/<ステップ数>/pretrained_model/（last は最新へのリンク）
# 途中で止まった場合の再開:
#   bash scripts/train_smolvla.sh <データセット名> <実行名> --resume=true \
#       --config_path=outputs/train/<実行名>/checkpoints/last/pretrained_model/train_config.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-pot_multi_v1}"
RUN="${2:-smolvla_${DATASET}}"
shift $(( $# < 2 ? $# : 2 ))
BASE="${SMOLVLA_BASE:-lerobot/smolvla_base}"

if [ ! -f "data/$DATASET/meta/info.json" ]; then
    echo "データセット data/$DATASET が見つかりません。" >&2
    exit 1
fi
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "GPU が使えません。GPU インスタンス（例: ml.g5.xlarge）で実行してください。" >&2
    exit 1
fi
RESUMING=0
for arg in "$@"; do [ "$arg" = "--resume=true" ] && RESUMING=1; done
if [ -e "outputs/train/$RUN" ] && [ "$RESUMING" = "0" ]; then
    echo "outputs/train/$RUN は既にあります。別の実行名にするか、--resume=true で再開してください。" >&2
    exit 1
fi

# GPU の数に合わせて、起動のしかた・1 GPU あたりのバッチ・データ読み込みの並列数を決める
NGPU="$("$PY" -c "import torch; print(torch.cuda.device_count())")"
BATCH=$(( 64 / NGPU > 0 ? 64 / NGPU : 1 ))
LOADERS=$(( ($(nproc) - 1) / NGPU ))
LOADERS=$(( LOADERS < 1 ? 1 : (LOADERS > 6 ? 6 : LOADERS) ))
if [ "$NGPU" -gt 1 ]; then
    LAUNCH=("$VENV_DIR/bin/accelerate" launch --multi_gpu --num_processes="$NGPU" --num_machines=1
            --mixed_precision=no --dynamo_backend=no "$VENV_DIR/bin/lerobot-train")
else
    LAUNCH=("$VENV_DIR/bin/lerobot-train")
fi
echo "GPU ${NGPU} 個で学習します（1 GPU あたりのバッチ ${BATCH}、合計 $(( BATCH * NGPU ))）。"

MODEL_ARGS=()
if [ "$RESUMING" = "0" ]; then
    # 事前学習済みモデルのカメラ名を読み、学習データのカメラ名との対応を作る（数秒）
    RENAME="$("$PY" -m so101_food.vla.rename_map --policy "$BASE" --dataset "$DATASET")"
    MODEL_ARGS=(--policy.path="$BASE" --rename_map="$RENAME")
fi

# SmolVLA の設定（lerobot/smolvla_base の既定値のまま）:
# - 行動チャンク 50 ステップ（25Hz で 2 秒分）を予測し、50 ステップ実行してから次を予測する
# - 画像を読む部分と言語モデルは固定し、行動を出す部分（action expert）だけを学習する
# - 画像は 512×512 に拡大して入力する。指示文（タスクの説明）も入力する
# - 合計バッチ 64 × 20,000 ステップ（lerobot の SmolVLA の手順と同じ。サンプル数の合計は ACT と同じ 128 万）
#   試運転（A10G 1 個）ではバッチ 32 で 1 ステップ 0.63 秒、GPU メモリ 7.7GB。
#   ml.g5.xlarge（A10G × 1）で約 7 時間、ml.g5.12xlarge（A10G × 4）で 2〜2.5 時間の見込み
run_in_background "train_${RUN}" \
    "${LAUNCH[@]}" \
    "${MODEL_ARGS[@]}" \
    --dataset.repo_id="local/$DATASET" \
    --dataset.root="$REPO_DIR/data/$DATASET" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --batch_size="$BATCH" \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers="$LOADERS" \
    --output_dir="outputs/train/$RUN" \
    --job_name="$RUN" \
    "$@"
