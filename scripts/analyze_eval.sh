#!/usr/bin/env bash
# 評価結果をまとめて比較する（どのインスタンスでも可。すぐ終わる）.
#
#   bash scripts/analyze_eval.sh <評価名...> [--dataset <学習データ名>]
#   例: bash scripts/analyze_eval.sh act_pot_multi_v1 act_pot_multi_v1_seen --dataset pot_multi_v1
#
# 各評価（outputs/eval/<評価名>/episodes.csv）の成功率と失敗の内訳を、掴む片までの距離ごと・
# 机の上の片の数ごとに表にする。--dataset を付けると、学習データで掴んだ片までの距離の分布も並べる。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

"$PY" -m so101_food.eval.analyze "$@"
