#!/usr/bin/env bash
# 学習データを生成する（CPU インスタンス向け。vCPU の数だけ並列に動かす）.
#
#   bash scripts/generate_dataset.sh [データセット名] [成功エピソード数]
#   例: bash scripts/generate_dataset.sh pot_cube_v1 300
#
# 出力: data/<データセット名>/（LeRobot v3 形式）と generation_stats.json
# 同名のデータセットを作り直すときは、3 番目以降に --overwrite を付ける。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NAME="${1:-pot_cube_v1}"
EPISODES="${2:-300}"
shift $(( $# < 2 ? $# : 2 ))

run_in_background "generate_${NAME}" \
    "$PY" -m so101_food.datagen.generate --name "$NAME" --episodes "$EPISODES" "$@"
