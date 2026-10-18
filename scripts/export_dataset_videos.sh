#!/usr/bin/env bash
# 生成したデータセットの中身を、見やすい動画に書き出す（どのインスタンスでも可。数分で終わる）.
#
#   bash scripts/export_dataset_videos.sh [データセット名] [エピソード数]
#   例: bash scripts/export_dataset_videos.sh pot_pieces_v1 6
#
# 出力: outputs/dataset_videos/<データセット名>/
#   ep<番号>_<物体名>.mp4 … 1 エピソードずつ（左: 俯瞰カメラ、右: 手首カメラ）
#   grid_9.mp4           … 9 エピソードの俯瞰カメラを 3×3 に並べた一覧
# 特定のエピソードを見たいときは --episodes 0 10 20 のように付ける。
# 動画は notebooks/view_videos.ipynb で再生できる。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NAME="${1:-pot_pieces_v1}"
NUM="${2:-6}"
shift $(( $# < 2 ? $# : 2 ))

"$PY" -m so101_food.datagen.export_videos --dataset "$NAME" --num "$NUM" "$@"
