#!/usr/bin/env bash
# 同じ配置で指示文の食材だけを変えてポリシーを動かすデモ動画を作る（段階3。CPU インスタンス向け）.
#
#   bash scripts/demo_language.sh [実行名] [配置の数] [追加オプション...]
#   例: bash scripts/demo_language.sh smolvla_pot_lang_v1 3
#       bash scripts/demo_language.sh smolvla_pot_lang_v1 3 --noise zero --name smolvla_pot_lang_v1_zero
#
# 学習の最新チェックポイント（outputs/train/<実行名>/checkpoints/last）を使う。
# 出力: outputs/demo/<実行名>/scene<番号>_seed<seed>.mp4（指示文 3 通りを続けて再生）, summary.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN="${1:-smolvla_pot_lang_v1}"
SCENES="${2:-3}"
shift $(( $# < 2 ? $# : 2 ))

POLICY="outputs/train/$RUN/checkpoints/last/pretrained_model"
if [ ! -d "$POLICY" ] && [[ " $* " != *" --policy "* ]]; then
    echo "$POLICY が見つかりません。学習が終わっているか確認してください。" >&2
    exit 1
fi

run_in_background "demo_${RUN}" \
    "$PY" -m so101_food.eval.demo_language --policy "$POLICY" --name "$RUN" --scenes "$SCENES" "$@"
