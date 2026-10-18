"""描画品質のプリセットと動画保存.

so101_nexus の既定品質（MSAA 8x・影 8192px）は OSMesa（CPU 描画）では
1 ステップ数百 ms かかるため、バックエンドに応じて品質を切り替える。

注意: 学習データ生成と評価では同じ品質で描画すること（見た目が変わると
ポリシーにとって分布外になる）。データ生成用の品質は段階1で固定する。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from so101_nexus import RenderingQuality

    from so101_food.utils.gl import GLInfo


def quality_preset(name: str) -> RenderingQuality:
    """``"fast"``（MSAA なし・影なし）または ``"high"``（so101_nexus 既定）を返す."""
    from so101_nexus import RenderingQuality

    if name == "fast":
        return RenderingQuality(msaa_samples=0, shadows=False, shadow_map_size=1024)
    if name == "high":
        return RenderingQuality()
    raise ValueError(f"unknown quality preset: {name!r}")


def quality_for_gl(gl: GLInfo) -> RenderingQuality:
    """GPU で描画できるなら高品質、CPU 描画なら高速プリセットを返す."""
    return quality_preset("high" if gl.hardware else "fast")


def save_mp4(frames: Sequence[np.ndarray], path: str | Path, fps: int = 25) -> Path:
    """RGB フレーム列を H.264 の mp4 として保存し、保存先を返す."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # yuv420p は偶数サイズが必要なので、奇数サイズは 1px 切り詰める
    h, w = frames[0].shape[:2]
    frames = [f[: h - h % 2, : w - w % 2] for f in frames]
    imageio.mimwrite(
        path, frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1
    )
    return path
