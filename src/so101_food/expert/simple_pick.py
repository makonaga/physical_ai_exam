"""直方体の小片を真上から掴んで持ち上げる、最小構成のスクリプト動作.

段階0で「SO-101 が 2〜3cm 角の小片を挟めるか」を確かめるためのもの。
鍋への投入までを含むエキスパートは :mod:`so101_food.expert.pot_expert`。

動作は関節空間の経由点を線形補間した目標関節角度（``pd_joint_pos`` の行動）:
上空へ移動（指を開く）→ 下降 → 指を閉じる → 持ち上げ → 保持
"""

from __future__ import annotations

import numpy as np

from so101_food.expert.ik import IKResult, TopDownIK
from so101_food.expert.trajectory import Segment, iter_actions, yaw_from_quat


def plan_top_down_pick(
    ik: TopDownIK,
    obj_pos: np.ndarray,
    obj_yaw: float,
    *,
    approach_height: float = 0.06,
    lift_height: float = 0.07,
    open_deg: float = 35.0,
    close_deg: float = -10.0,
) -> tuple[list[Segment], IKResult]:
    """直方体を真上から掴んで持ち上げる区間列と、把持姿勢の IK 結果を返す.

    閉じ方向は直方体の面に垂直な 2 軸から、手首のひねりが小さいものを選ぶ。
    ``obj_pos`` は小片の中心で、TCP をその高さまで下ろす。真上姿勢を保てるのは
    TCP の高さがおよそ 9cm までなので（手首ピッチの可動域の限界）、上昇量は控えめにする。
    """
    obj_pos = np.asarray(obj_pos, dtype=float)
    grasp = ik.solve_best_yaw(obj_pos, [obj_yaw, obj_yaw + np.pi / 2])
    pre = ik.solve(obj_pos + [0.0, 0.0, approach_height], grasp.yaw, grasp.q)
    lift = ik.solve(obj_pos + [0.0, 0.0, lift_height], grasp.yaw, grasp.q)
    g_open, g_close = np.radians(open_deg), np.radians(close_deg)
    segments = [
        Segment("approach", pre.q, g_open, 40),
        Segment("descend", grasp.q, g_open, 30),
        Segment("close", grasp.q, g_close, 20),
        Segment("lift", lift.q, g_close, 40),
        Segment("hold", lift.q, g_close, 20),
    ]
    return segments, grasp


def run_pick_trial(env, ik: TopDownIK, seed: int, render: bool = False) -> dict:
    """``MuJoCoPickLift-v1``（``pd_joint_pos``）で 1 回だけ把持を試し、結果を返す.

    ``render=True`` なら ``env.render()`` のフレームを ``frames`` に入れて返す。
    """
    env.reset(seed=seed)
    u = env.unwrapped
    pose = u._get_target_pose()
    segments, grasp = plan_top_down_pick(ik, pose[:3], yaw_from_quat(pose[3:]))
    frames = []
    info = {}
    for _, action in iter_actions(segments, u._get_current_qpos()):
        _, _, _, _, info = env.step(action)
        if render:
            frames.append(env.render())
    return {
        "seed": seed,
        "obj_xy": pose[:2].round(3).tolist(),
        "ik_pos_err_mm": round(grasp.pos_err_m * 1e3, 2),
        "lift_height_m": round(float(info["lift_height"]), 3),
        "is_grasped": bool(info["is_grasped"] > 0.5),
        "success": bool(info["success"]),
        "frames": frames,
    }
