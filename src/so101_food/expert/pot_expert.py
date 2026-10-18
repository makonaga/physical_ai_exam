"""小片を掴んで鍋に入れるスクリプテッド・エキスパート.

シミュレータから小片の真の姿勢を読み、関節空間の経由点を IK で求めて補間する。

1. 上空へ移動しながら指を開く（真上把持）
2. 下降して小片の中心の高さに TCP を合わせる
3. 指を閉じる
4. 持ち上げる
5. 鍋の上へ運ぶ（真上姿勢は高さ約 9cm までしか保てないので、ここでは傾きを許す）
6. 指を開いて落とす
7. 開始時の姿勢に戻る

学習データの多様性のため、各区間の時間・高さ・落とす位置をエピソードごとに少し揺らす
（乱数はエピソードの seed から作るので再現できる）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from so101_food.env.pot_task import CONTROL_HZ, FRAME_SKIP, PotTaskEnv, failure_reason
from so101_food.expert.ik import TopDownIK
from so101_food.expert.trajectory import Segment, iter_actions, yaw_from_quat

DEFAULT_DURATIONS = {  # 各区間の長さ [秒]
    "approach": 1.2,
    "descend": 0.8,
    "close": 0.5,
    "lift": 0.6,
    "carry": 1.4,
    "release": 0.5,
    "home": 1.4,
    "settle": 0.4,
}


@dataclass(frozen=True)
class ExpertParams:
    open_deg: float = 35.0  # 開いたときのグリッパ角度（パッド間 約 57mm）
    close_deg: float = -10.0  # 閉じる目標（小片に当たって止まり、握る力になる）
    approach_height: tuple[float, float] = (0.05, 0.07)  # 小片の上空の高さ（一様乱数の範囲）
    lift_height: float = 0.07  # 持ち上げ後の TCP の高さ（小片の中心から）
    carry_clearance: tuple[float, float] = (0.06, 0.08)  # 鍋の縁から TCP までの高さ
    drop_xy_noise: float = 0.01  # 落とす位置のばらつき（鍋の中心から）
    time_jitter: float = 0.15  # 区間の長さを ±15% 揺らす
    durations: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DURATIONS))


@dataclass
class Plan:
    segments: list[Segment]
    grasp_yaw: float
    ik_ok: bool


class PotExpert:
    def __init__(self, env: PotTaskEnv, params: ExpertParams | None = None) -> None:
        self.env = env
        self.params = params if params is not None else ExpertParams()
        self.grasp_ik = TopDownIK(env.model)
        # 鍋の上では位置を優先し、姿勢は「なるべく下向き」程度にする
        self.carry_ik = TopDownIK(env.model, rot_weight=0.02)

    def plan(self, rng: np.random.Generator) -> Plan:
        """現在の状態（reset 直後）から、エピソード全体の区間列を作る."""
        p, env = self.params, self.env
        q0 = env.robot_qpos()
        pose = env.object_pose()
        obj_pos, obj_yaw = pose[:3], yaw_from_quat(pose[3:])

        grasp = self.grasp_ik.solve_best_yaw(obj_pos, [obj_yaw, obj_yaw + np.pi / 2])
        # 真上姿勢がとれる高さはベースからの距離で変わるので、解けなければ上空の高さを下げる
        for height in (rng.uniform(*p.approach_height), 0.05, 0.04, 0.03):
            pre = self.grasp_ik.solve(obj_pos + [0.0, 0.0, height], grasp.yaw, grasp.q)
            if pre.ok():
                break
        lift = self.grasp_ik.solve(obj_pos + [0.0, 0.0, p.lift_height], grasp.yaw, grasp.q)
        cx, cy = env.pot.center
        drop = np.array(
            [
                cx + rng.uniform(-p.drop_xy_noise, p.drop_xy_noise),
                cy + rng.uniform(-p.drop_xy_noise, p.drop_xy_noise),
                env.pot.height + rng.uniform(*p.carry_clearance),
            ]
        )
        carry = self.carry_ik.solve(drop, grasp.yaw, lift.q)
        ik_ok = grasp.ok() and pre.ok() and carry.pos_err_m < 0.005

        g_open, g_close = np.radians(p.open_deg), np.radians(p.close_deg)
        targets = [
            ("approach", pre.q, g_open),
            ("descend", grasp.q, g_open),
            ("close", grasp.q, g_close),
            ("lift", lift.q, g_close),
            ("carry", carry.q, g_close),
            ("release", carry.q, g_open),
            ("home", q0[:5], q0[5]),
            ("settle", q0[:5], q0[5]),
        ]
        segments = []
        for name, q_arm, gripper in targets:
            seconds = p.durations[name] * rng.uniform(1 - p.time_jitter, 1 + p.time_jitter)
            segments.append(Segment(name, q_arm, gripper, max(1, round(seconds * CONTROL_HZ))))
        return Plan(segments=segments, grasp_yaw=grasp.yaw, ik_ok=ik_ok)


@dataclass
class EpisodeRecord:
    """1 エピソード分の記録（ポリシーの周期 25Hz）.

    ``qpos[t]`` は行動 ``action[t]`` を出す直前の全関節の状態（画像の再描画に使う）、
    ``state[t]`` はそのときのロボット 6 関節の角度 [rad]、``action[t]`` は目標角度 [rad]。
    """

    seed: int
    qpos: np.ndarray
    state: np.ndarray
    action: np.ndarray
    reason: str
    success: bool
    obj_pose0: np.ndarray
    info: dict

    @property
    def n_frames(self) -> int:
        return len(self.action)


def run_expert_episode(env: PotTaskEnv, expert: PotExpert, seed: int) -> EpisodeRecord:
    """seed で reset し、エキスパートを最後まで実行して記録を返す（描画はしない）."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obj_pose0 = env.object_pose()
    plan = expert.plan(rng)
    qpos, state, action = [], [], []
    info: dict = {}
    for _, act in iter_actions(plan.segments, env.robot_qpos(), ease=True):
        qpos.append(env.data.qpos.copy())
        state.append(env.robot_qpos())
        action.append(act)
        for _ in range(FRAME_SKIP):
            _, _, _, _, info = env.step(act)
    reason = failure_reason(info) if plan.ik_ok else "ik_failed"
    return EpisodeRecord(
        seed=seed,
        qpos=np.asarray(qpos),
        state=np.asarray(state, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        reason=reason,
        success=reason == "success",
        obj_pose0=obj_pose0,
        info=info,
    )
