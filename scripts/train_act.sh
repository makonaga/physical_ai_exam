#!/usr/bin/env bash
# ACT を学習する（GPU インスタンス向け）.
#
#   bash scripts/train_act.sh [データセット名] [実行名] [lerobot-train の追加オプション...]
#   例: bash scripts/train_act.sh pot_cube_v1 act_pot_cube_v1
#       bash scripts/train_act.sh pot_cube_v1 act_test --steps=2000   # 短い試運転
#
# 出力: outputs/train/<実行名>/checkpoints/<ステップ数>/pretrained_model/
#      （last は最新のチェックポイントへのリンク）
# 途中で止まった場合の再開:
#   bash scripts/train_act.sh <データセット名> <実行名> --resume=true \
#       --config_path=outputs/train/<実行名>/checkpoints/last/pretrained_model/train_config.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${1:-pot_cube_v1}"
RUN="${2:-act_${DATASET}}"
shift $(( $# < 2 ? $# : 2 ))

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

# ACT の設定:
# - 行動チャンク 50 ステップ（25Hz で 2 秒分）を予測し、50 ステップ実行してから次を予測する
# - 学習データはおよそ 300 エピソード × 170 フレーム ≒ 5 万フレーム
run_in_background "train_${RUN}" \
    "$VENV_DIR/bin/lerobot-train" \
    --dataset.repo_id="local/$DATASET" \
    --dataset.root="$REPO_DIR/data/$DATASET" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --wandb.enable=false \
    --batch_size=16 \
    --steps=40000 \
    --save_freq=10000 \
    --log_freq=200 \
    --num_workers=3 \
    --output_dir="outputs/train/$RUN" \
    --job_name="$RUN" \
    "$@"
