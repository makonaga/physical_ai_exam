"""関節空間の経由点を補間して、目標関節角度の列（``pd_joint_pos`` の行動）を作る."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """``n_steps`` かけて腕とグリッパの目標を ``q_arm`` / ``gripper`` へ動かす区間."""

    name: str
    q_arm: np.ndarray
    gripper: float
    n_steps: int


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """クォータニオン（w, x, y, z）から鉛直軸まわりの回転角を取り出す."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def iter_actions(
    segments: list[Segment], q_start: np.ndarray, ease: bool = False
) -> Iterator[tuple[str, np.ndarray]]:
    """区間列を補間し、(区間名, 6 次元の目標関節角度) を 1 ステップずつ返す.

    ``q_start`` は開始時の 6 関節（腕 5 + グリッパ）の角度。``ease=True`` なら
    区間の始めと終わりで速度が 0 になる補間（余弦）にして、区間の切り替わりでの
    急な速度変化をなくす。
    """
    prev = np.asarray(q_start, dtype=float)
    for seg in segments:
        goal = np.concatenate([seg.q_arm, [seg.gripper]])
        for i in range(1, seg.n_steps + 1):
            a = i / seg.n_steps
            if ease:
                a = 0.5 - 0.5 * np.cos(np.pi * a)
            yield seg.name, ((1.0 - a) * prev + a * goal).astype(np.float32)
        prev = goal
