#!/usr/bin/env bash
# 学習の進み具合（ログの最新の数行）を、数分ごとに S3 にコピーする（手元のパソコンから確認するため）.
#
#   bash scripts/sync_progress_s3.sh <実行名> <S3 のコピー先> [間隔（秒）]
#   例: bash scripts/sync_progress_s3.sh smolvla_pot_multi_v1 s3://<バケット名>/physical_ai
#
# <コピー先>/<実行名>_progress.txt に、時刻・学習中かどうか・ログの最新の step 行（3 行）を書く。
# 最初の 1 回はその場で書き（S3 の権限エラーがあればすぐ分かる）、そのあとは裏で 5 分ごとに書き直す。
# 学習が終わると、最後に 1 回書いてから止まる。手元のパソコンでは次で見る（AWS CLI の設定が必要）:
#   aws s3 cp s3://<バケット名>/physical_ai/<実行名>_progress.txt -
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ $# -lt 2 ]; then
    echo "使い方: bash scripts/sync_progress_s3.sh <実行名> <S3 のコピー先（s3://...）> [間隔（秒）]" >&2
    exit 1
fi
RUN="$1"
DEST="${2%/}/${RUN}_progress.txt"
INTERVAL="${3:-300}"
LOG="$(ls -t logs/train_"${RUN}"_*.log 2>/dev/null | head -n 1)"
if [ -z "$LOG" ]; then
    echo "logs/train_${RUN}_*.log がありません。学習を始めてから実行してください。" >&2
    exit 1
fi
TMP="/tmp/${RUN}_progress.txt"

training() { pgrep -f "lerobot-train.*outputs/train/${RUN}( |$)" > /dev/null; }

upload() {
    {
        echo "$(date '+%Y-%m-%d %H:%M:%S')  $(basename "$LOG")  $(training && echo 学習中 || echo 学習は終わっています)"
        grep "step:" "$LOG" | tail -n 3 | sed 's/.*\r//' || true
    } > "$TMP"
    aws s3 cp --quiet "$TMP" "$DEST"
}

if [ "${SYNC_LOOP:-0}" = "1" ]; then
    # 裏で動く部分: 学習中は INTERVAL 秒ごとに書き直し、終わったら最後に 1 回書いて止まる
    while training; do
        sleep "$INTERVAL"
        upload || true
    done
    upload
    exit 0
fi

upload
echo "$DEST に書きました。手元のパソコンでは次で確認できます:"
echo "  aws s3 cp $DEST -"
SYNC_LOOP=1 run_in_background "sync_${RUN}" env SYNC_LOOP=1 bash "$0" "$@"
