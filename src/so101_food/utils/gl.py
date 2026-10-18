"""MuJoCo のオフスクリーン描画バックエンドを選ぶ.

MuJoCo は import 時に ``MUJOCO_GL`` を読むため、選択は MuJoCo（および
so101_nexus）を import する前に行う必要がある。描画の可否はプロセス単位で
決まるので、候補ごとに子プロセスで試し描画して判定する。

- ``egl``    : ヘッドレス描画。NVIDIA の EGL が使えれば GPU で描画する
- ``osmesa`` : CPU のソフトウェア描画（遅いが GPU なしでも動く）

``egl`` で試し描画に成功しても、NVIDIA の EGL が見えない環境では Mesa の
ソフトウェア描画（llvmpipe）で動いていることがある。そのため GL_RENDERER の
文字列も調べ、GPU で描画しているか（``hardware``）を返す。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

_PROBE_SCRIPT = """
import mujoco
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 1"/><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 32, 32)
renderer.update_scene(data)
assert renderer.render().sum() > 0
try:
    from OpenGL import GL
    print("renderer:", GL.glGetString(GL.GL_RENDERER).decode(errors="replace"))
except Exception:
    print("renderer: unknown")
"""

_SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "swiftshader", "software")


@dataclass(frozen=True)
class GLInfo:
    backend: str  # MUJOCO_GL に設定した値
    renderer: str  # OpenGL の GL_RENDERER 文字列
    hardware: bool  # GPU で描画しているか


def probe_gl_backend(backend: str, timeout: float = 60.0) -> GLInfo | str:
    """子プロセスで ``backend`` による試し描画を行う. 成功なら GLInfo、失敗なら理由を返す."""
    env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout:.0f}s"
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return lines[-1] if lines else f"exit code {proc.returncode}"
    renderer = "unknown"
    for line in proc.stdout.splitlines():
        if line.startswith("renderer:"):
            renderer = line.removeprefix("renderer:").strip()
    hardware = renderer != "unknown" and not any(s in renderer.lower() for s in _SOFTWARE_RENDERERS)
    return GLInfo(backend=backend, renderer=renderer, hardware=hardware)


def select_gl_backend(
    candidates: tuple[str, ...] = ("egl", "osmesa"), verbose: bool = True
) -> GLInfo:
    """描画できるバックエンドを選んで ``MUJOCO_GL`` に設定し、その情報を返す.

    GPU で描画できるものを優先し、なければ最初に描画できたものを使う。
    環境変数 ``MUJOCO_GL`` が既に設定されていれば、それを最優先で試す。
    """
    if "mujoco" in sys.modules:
        raise RuntimeError(
            "mujoco は既に import されています。select_gl_backend() は mujoco / so101_nexus "
            "の import より前に呼んでください（ノートブックならカーネルを再起動）。"
        )
    preset = os.environ.get("MUJOCO_GL")
    order = ([preset] if preset else []) + [c for c in candidates if c != preset]
    found: list[GLInfo] = []
    for backend in order:
        result = probe_gl_backend(backend)
        if verbose:
            if isinstance(result, GLInfo):
                kind = "GPU" if result.hardware else "CPU（ソフトウェア描画）"
                print(f"[gl] {backend:7s}: OK  {result.renderer} → {kind}")
            else:
                print(f"[gl] {backend:7s}: NG  {result}")
        if isinstance(result, GLInfo):
            found.append(result)
            if result.hardware:
                break
    if not found:
        raise RuntimeError(
            f"どの描画バックエンドも使えません: {order}. scripts/setup_sagemaker.sh で "
            "OS パッケージ（libegl1, libosmesa6 など）を入れてください。"
        )
    chosen = next((g for g in found if g.hardware), found[0])
    os.environ["MUJOCO_GL"] = chosen.backend
    os.environ["PYOPENGL_PLATFORM"] = chosen.backend
    return chosen
