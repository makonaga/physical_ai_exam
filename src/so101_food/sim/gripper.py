"""SO-101 グリッパの開口幅を MJCF から実測する.

SO-101 の可動側の指はヒンジで回転するため、開口幅は関節角度と
指先からの深さで変わる。ここでは次の 2 つを関節角度ごとに測る。

- ``tip_gap_mm`` : 両指先（先端の球ジオム）の中心間距離
- ``pad_gap_mm`` : 指の内側パッド（先端寄りの箱ジオム）同士の最短距離。
  実際に挟める物体幅の目安になる
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

FIXED_TIP = "fixed_jaw_sph_tip1"
MOVING_TIP = "moving_jaw_sph_tip1"
FIXED_PADS = ("fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6")
MOVING_PADS = ("moving_jaw_box2", "moving_jaw_box3")
TCP_SITE = "gripperframe"


@dataclass(frozen=True)
class GripperSample:
    angle_deg: float
    tip_gap_mm: float
    pad_gap_mm: float


def measure_gripper(
    model: mujoco.MjModel, angles_deg: np.ndarray | None = None
) -> list[GripperSample]:
    """グリッパ関節角度ごとの開口幅を測る（``model`` の状態は変えない）."""
    joint = model.joint("gripper")
    lo, hi = np.degrees(model.jnt_range[joint.id])
    if angles_deg is None:
        angles_deg = np.append(np.arange(np.ceil(lo), hi, 5.0), hi)
    data = mujoco.MjData(model)
    tip_f, tip_m = model.geom(FIXED_TIP).id, model.geom(MOVING_TIP).id
    pairs = [(model.geom(a).id, model.geom(b).id) for a in FIXED_PADS for b in MOVING_PADS]
    samples = []
    for deg in angles_deg:
        data.qpos[joint.qposadr[0]] = np.radians(deg)
        mujoco.mj_forward(model, data)
        tip_gap = np.linalg.norm(data.geom_xpos[tip_f] - data.geom_xpos[tip_m])
        pad_gap = min(mujoco.mj_geomDistance(model, data, a, b, 0.5, None) for a, b in pairs)
        samples.append(GripperSample(float(deg), float(tip_gap * 1e3), float(pad_gap * 1e3)))
    return samples


def angle_for_gap(samples: list[GripperSample], gap_mm: float) -> float:
    """パッド間距離が ``gap_mm`` になるグリッパ角度 [deg] を線形補間で返す."""
    angles = np.array([s.angle_deg for s in samples])
    gaps = np.array([s.pad_gap_mm for s in samples])
    if not gaps[0] <= gap_mm <= gaps[-1]:
        raise ValueError(f"gap {gap_mm} mm is outside the measurable range")
    return float(np.interp(gap_mm, gaps, angles))


def tcp_to_fingertip_mm(model: mujoco.MjModel) -> float:
    """TCP（gripperframe サイト）から固定側指先までの距離 [mm]."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    tcp = data.site_xpos[model.site(TCP_SITE).id]
    tip = data.geom_xpos[model.geom(FIXED_TIP).id]
    return float(np.linalg.norm(tcp - tip) * 1e3)


def samples_as_dicts(samples: list[GripperSample]) -> list[dict]:
    return [asdict(s) for s in samples]
