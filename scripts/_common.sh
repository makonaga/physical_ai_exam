# scripts/*.sh から読み込む共通処理（単体では実行しない）.
#
# run_in_background <ログ名> <コマンド...>
#   コマンドを nohup で裏で動かし、ログを logs/<ログ名>_<日時>.log に書く。
#   ブラウザやターミナルを閉じても処理は続く。FOREGROUND=1 なら前面で実行する。

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/pai}"
PY="$VENV_DIR/bin/python"
cd "$REPO_DIR"

if [ ! -x "$PY" ]; then
    echo "仮想環境 $VENV_DIR がありません。先に bash scripts/setup_sagemaker.sh を実行してください。" >&2
    exit 1
fi

run_in_background() {
    local name="$1"
    shift
    if [ "${FOREGROUND:-0}" = "1" ]; then
        "$@"
        return
    fi
    mkdir -p logs
    local log="logs/${name}_$(date +%Y%m%d_%H%M%S).log"
    nohup "$@" > "$log" 2>&1 &
    local pid=$!
    echo "裏で実行を始めました（PID $pid）。"
    echo "  進み具合を見る : tail -f $REPO_DIR/$log    （見るのをやめるときは Ctrl+C。処理は止まりません）"
    echo "  処理を止める   : kill $pid"
}
