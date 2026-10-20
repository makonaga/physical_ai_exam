#!/usr/bin/env bash
# 生成済みの学習データを 1 つにまとめる（元のデータは残す）.
#
#   bash scripts/merge_datasets.sh <まとめたデータの名前> <元のデータ名...>
#   例: bash scripts/merge_datasets.sh pot_lang_v2 pot_lang_v1 pot_lang_v1b
#
# 出力: data/<まとめたデータの名前>/（LeRobot v3 形式）, episodes_meta.jsonl, generation_stats.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ $# -lt 3 ]; then
    echo "使い方: bash scripts/merge_datasets.sh <まとめたデータの名前> <元のデータ名> <元のデータ名> ..." >&2
    exit 1
fi
NAME="$1"
shift

run_in_background "merge_${NAME}" "$PY" -m so101_food.datagen.merge --name "$NAME" "$@"
