"""任意の凸形状を真上から掴むための把持点・向きを、頂点から求める.

上から見た形（頂点を水平面に投影した凸包）で、幅が最も狭くなる方向を指の閉じる
方向にする（回転キャリパー法: 凸多角形の最小幅は、ある辺の法線方向で実現される）。
掴む位置は、その方向とそれに直交する方向それぞれで形の中央、高さは小片の上下の中央。
立方体なら面の法線方向・中心・半分の高さになり、段階1の掴み方と一致する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class GraspTarget:
    center: np.ndarray  # TCP の目標位置（世界座標）[m]
    yaw: float  # 指の閉じる方向（水平面内の角度）[rad]
    width: float  # 閉じる方向の小片の幅 [m]
    height: float  # 小片の高さ [m]
    yaw_candidates: tuple[float, ...]  # 同じくらい細い向きも含めた閉じる方向の候補


def plan_top_down_grasp(
    vertices: np.ndarray, min_tcp_height: float = 0.010, width_tolerance: float = 0.004
) -> GraspTarget:
    """小片の頂点（世界座標、(N, 3)）から真上把持の目標を求める.

    ``min_tcp_height``: 低い小片でも指先が床に当たらないよう、TCP の高さの下限を床から取る。
    ``width_tolerance``: 直交する方向の幅が最小幅とこれ以内の差なら、その向きも候補にする
    （立方体のように 2 方向がほぼ同じ幅なら、手首のひねりが少ない方を IK で選べる）。
    """
    xy = vertices[:, :2]
    polygon = xy[ConvexHull(xy).vertices]
    best_width, best_u = np.inf, None
    for i in range(len(polygon)):
        edge = polygon[(i + 1) % len(polygon)] - polygon[i]
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        u = np.array([-edge[1], edge[0]]) / length
        width = float(np.ptp(polygon @ u))
        if width < best_width:
            best_width, best_u = width, u
    v = np.array([-best_u[1], best_u[0]])
    pu, pv = polygon @ best_u, polygon @ v
    center_xy = best_u * (pu.min() + pu.max()) / 2 + v * (pv.min() + pv.max()) / 2
    z_lo, z_hi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    z = max((z_lo + z_hi) / 2, z_lo + min_tcp_height)
    yaw = float(np.arctan2(best_u[1], best_u[0]))
    candidates = (yaw, yaw + np.pi / 2) if np.ptp(pv) - best_width < width_tolerance else (yaw,)
    return GraspTarget(
        center=np.array([center_xy[0], center_xy[1], z]),
        yaw=yaw,
        width=best_width,
        height=z_hi - z_lo,
        yaw_candidates=candidates,
    )
