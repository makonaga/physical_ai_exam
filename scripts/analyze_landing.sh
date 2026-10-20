#!/usr/bin/env bash
# 評価で手先を下ろした位置が、机の上のどの片に一番近かったかを調べる（複数の片を置く評価向け。CPU で数分）.
#
#   bash scripts/analyze_landing.sh <評価名...>
#   例: bash scripts/analyze_landing.sh smolvla_pot_lang_v1 act_pot_lang_v1 smolvla_pot_multi_v1
#
# 出力: 画面と outputs/eval/<評価名>/landing.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

"$PY" -m so101_food.eval.landing "$@"
