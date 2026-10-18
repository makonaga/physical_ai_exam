#!/usr/bin/env bash
# SageMaker Studio（JupyterLab スペース）用のセットアップ.
#
#   cd ~/physical_ai_exam && bash scripts/setup_sagemaker.sh
#
# - OS パッケージ（EGL / OSMesa / ffmpeg）はホーム外に入るため、スペースを再起動すると
#   消える。スペースを起動するたびにこのスクリプトを実行すること（2 回目以降は短時間で終わる）。
# - Python 環境は ~/.venvs/pai（ホーム配下 = EBS に永続化）に作り、
#   Jupyter カーネル「Python 3.12 (pai)」として登録する。
# - PyTorch の CUDA 版は GPU ドライバのバージョンから自動で選ぶ。
#   手動指定: TORCH_BACKEND=cu126 bash scripts/setup_sagemaker.sh （cpu / cu118 / cu126 / cu128 / cu130）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/pai}"
KERNEL_NAME="pai"
KERNEL_DISPLAY="Python 3.12 (pai)"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 1. OS パッケージ
log "1/5 OS パッケージ（libegl1 libgl1 libglx-mesa0 libosmesa6 ffmpeg）"
if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libegl1 libgl1 libglx-mesa0 libosmesa6 ffmpeg >/dev/null
else
    echo "sudo が使えないため OS パッケージをスキップします（描画に失敗する場合は管理者に確認）。"
fi

# ---------------------------------------------------------------- 2. uv
log "2/5 uv（Python パッケージ管理）"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

# ---------------------------------------------------------------- 3. 仮想環境
log "3/5 仮想環境 $VENV_DIR（Python 3.12）"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    # イメージ内蔵の /opt/conda の Python ではなく、uv がホーム配下に入れる Python を使う
    # （スペースのイメージを更新しても仮想環境が壊れないようにするため）
    uv venv --python 3.12 --managed-python "$VENV_DIR"
fi

# ---------------------------------------------------------------- 4. PyTorch の CUDA 版を決めてインストール
if [ -z "${TORCH_BACKEND:-}" ]; then
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    if [ -z "$DRIVER" ]; then
        TORCH_BACKEND="cpu"
    else
        MAJOR="${DRIVER%%.*}"
        # CUDA 12.x / 13.x の各ビルドが動く最小ドライバ（Linux）: 13.0 → 580, 12.8 → 570, 12.x → 525
        if   [ "$MAJOR" -ge 580 ]; then TORCH_BACKEND="cu130"
        elif [ "$MAJOR" -ge 570 ]; then TORCH_BACKEND="cu128"
        elif [ "$MAJOR" -ge 525 ]; then TORCH_BACKEND="cu126"
        else TORCH_BACKEND="cu118"
        fi
    fi
fi
log "4/5 パッケージのインストール（PyTorch: $TORCH_BACKEND, GPU ドライバ: ${DRIVER:-なし/手動指定}）"

# 前回と異なる CUDA 版を選んだ場合（例: CPU インスタンスで作った環境を GPU で使う）は torch を入れ直す
REINSTALL=()
STAMP="$VENV_DIR/.torch_backend"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$TORCH_BACKEND" ]; then
    echo "PyTorch の CUDA 版が $(cat "$STAMP") → $TORCH_BACKEND に変わるため torch を入れ直します。"
    REINSTALL=(--reinstall-package torch --reinstall-package torchvision)
fi

uv pip install --python "$VENV_DIR/bin/python" --torch-backend "$TORCH_BACKEND" \
    ${REINSTALL[@]+"${REINSTALL[@]}"} -e "$REPO_DIR[notebook,dev]"
echo "$TORCH_BACKEND" > "$STAMP"

# ---------------------------------------------------------------- 5. Jupyter カーネル登録と確認
log "5/5 Jupyter カーネル「$KERNEL_DISPLAY」を登録"
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY" >/dev/null

"$VENV_DIR/bin/python" - <<'EOF'
import importlib.metadata as md
import torch
for pkg in ("so101-food", "so101-nexus", "lerobot", "mujoco", "torch"):
    print(f"  {pkg:12s} {md.version(pkg)}")
print(f"  CUDA 利用可能: {torch.cuda.is_available()}"
      + (f"（{torch.cuda.get_device_name(0)}）" if torch.cuda.is_available() else ""))
try:  # LeRobot が動画の読み込みに使う。torch とバージョンが合わないと import に失敗する
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    print("  torchcodec   OK")
except Exception as e:
    print(f"  torchcodec   NG（{type(e).__name__}）→ 学習時は --dataset.video_backend=pyav を指定")
EOF

log "完了。ノートブックでカーネル「$KERNEL_DISPLAY」を選んでください。"
