#!/usr/bin/env bash
# 指示文だけを変えたときにポリシーの動きがどれだけ変わるかを調べる（段階3。GPU があれば速い）.
#
#   bash scripts/probe_language.sh <実行名> <学習データ名> [追加オプション...]
#   例: bash scripts/probe_language.sh smolvla_pot_lang_v1 pot_lang_v1
#       bash scripts/probe_language.sh smolvla_pot_lang_v1 pot_lang_v1 --device cpu   # GPU がないとき
#
# 学習の最新チェックポイント（outputs/train/<実行名>/checkpoints/last）を使う。数分で終わるので裏では動かさない。
# 出力: 画面と outputs/probe/<実行名>.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN="$1"
DATASET="$2"
shift 2
"$PY" -m so101_food.eval.probe_language \
    --policy "outputs/train/$RUN/checkpoints/last/pretrained_model" --dataset "$DATASET" "$@"
