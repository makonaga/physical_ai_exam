#!/usr/bin/env bash
# 学習済みポリシーを評価する（CPU インスタンス向け。並列に動かす）.
#
#   bash scripts/evaluate.sh [実行名] [エピソード数] [追加オプション...]
#   例: bash scripts/evaluate.sh act_pot_cube_v1 50
#
# 学習の最新チェックポイント（outputs/train/<実行名>/checkpoints/last）を使う。
# 特定のチェックポイントを使う場合は --policy <pretrained_model のパス> --name <保存名> を付ける。
# 出力: outputs/eval/<実行名>/summary.json, episodes.csv, videos/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN="${1:-act_pot_cube_v1}"
EPISODES="${2:-50}"
shift $(( $# < 2 ? $# : 2 ))

POLICY="outputs/train/$RUN/checkpoints/last/pretrained_model"
if [ ! -d "$POLICY" ] && [[ " $* " != *" --policy "* ]]; then
    echo "$POLICY が見つかりません。学習が終わっているか確認してください。" >&2
    exit 1
fi

run_in_background "eval_${RUN}" \
    "$PY" -m so101_food.eval.evaluate --policy "$POLICY" --name "$RUN" --episodes "$EPISODES" "$@"
