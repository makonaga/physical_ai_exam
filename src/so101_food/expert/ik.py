"""真上から掴むための逆運動学（IK）.

SO-101 の腕は 5 自由度（肩パン・肩リフト・肘・手首ピッチ・手首ロール）。
肩パンで腕の平面を目標方向へ向け、リフト・肘・手首ピッチで平面内の位置と
「工具軸を真下に向ける」姿勢を作り、手首ロールで指の閉じる向き（ヨー）を
合わせる。手首ロールの可動域は約 ±160° だが、閉じ方向を向きのない軸として扱えば
（指の左右の入れ替えを許せば）任意のヨーで真上把持できる。ただし真上姿勢を
保てるのは TCP の高さがおよそ 9cm まで（手首ピッチの可動域 ±95° の限界）。

TCP は so101_nexus の ``gripperframe`` サイト（両指の間）。サイトの局所軸は
z = 手首から指先へ向かう方向、x = 指が閉じる方向の軸（可動指 → 固定指の向き）。
目標姿勢は z を世界の真下、x を水平面内のヨー角 ``yaw`` の方向に合わせる。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_SITE = "gripperframe"

# 「指先を下に向けて前方へ伸ばした」姿勢。肩パンは目標方向に合わせて上書きする
_REACH_DOWN_SEED_DEG = np.array([0.0, 0.0, 30.0, 70.0, 0.0])


def top_down_rotation(yaw: float) -> np.ndarray:
    """工具軸 z が真下、閉じる方向 x がヨー ``yaw`` を向く回転行列を返す."""
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # 腕 5 関節の角度 [rad]
    pos_err_m: float  # TCP 位置誤差
    tilt_deg: float  # 工具軸と真下のなす角
    yaw: float  # 目標にした閉じ方向のヨー [rad]

    def ok(self, pos_tol_m: float = 0.003, tilt_tol_deg: float = 5.0) -> bool:
        return self.pos_err_m < pos_tol_m and self.tilt_deg < tilt_tol_deg


class TopDownIK:
    """減衰最小二乗法で真上把持の関節角度を解く.

    ``model`` を共有し、計算には専用の ``MjData`` を使うので、
    シミュレーション中の状態には影響しない。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        rot_weight: float = 0.3,
        damping: float = 0.05,
        iterations: int = 150,
        max_step_rad: float = 0.2,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(TCP_SITE).id
        joints = [model.joint(name) for name in ARM_JOINTS]
        self.qpos_addr = np.array([j.qposadr[0] for j in joints])
        self.dof_addr = np.array([j.dofadr[0] for j in joints])
        self.q_low = np.array([model.jnt_range[j.id][0] for j in joints])
        self.q_high = np.array([model.jnt_range[j.id][1] for j in joints])
        self.rot_weight = rot_weight
        self.damping = damping
        self.iterations = iterations
        self.max_step_rad = max_step_rad
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        # 肩パンの正方向で TCP の方位角が増えるか減るか（SO-101 では減る）を順運動学で調べる
        probe = np.radians(_REACH_DOWN_SEED_DEG)
        probe[0] = 0.3
        p, _ = self.tcp_pose(probe)
        self._pan_sign = float(np.sign(np.arctan2(p[1], p[0])))

    def seed_for(self, target_pos: np.ndarray) -> np.ndarray:
        """目標位置に肩パンを向けた初期値を返す."""
        seed = np.radians(_REACH_DOWN_SEED_DEG)
        seed[0] = self._pan_sign * np.arctan2(target_pos[1], target_pos[0])
        return np.clip(seed, self.q_low, self.q_high)

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """関節角度 ``q`` での TCP の位置と回転行列を返す."""
        self.data.qpos[self.qpos_addr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def solve(
        self, target_pos: np.ndarray, yaw: float, q_seed: np.ndarray | None = None
    ) -> IKResult:
        """TCP を ``target_pos`` に、閉じ方向をヨー ``yaw`` に合わせる関節角度を解く."""
        target_pos = np.asarray(target_pos, dtype=float)
        q = self.seed_for(target_pos) if q_seed is None else np.array(q_seed, dtype=float)
        r_target = top_down_rotation(yaw)
        quat_err = np.zeros(4)
        rot_err = np.zeros(3)
        w = self.rot_weight
        for _ in range(self.iterations):
            self.data.qpos[self.qpos_addr] = q
            mujoco.mj_kinematics(self.model, self.data)
            # mj_jacSite は重心関連の量を参照するため mj_comPos で更新しておく
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            r_now = self.data.site_xmat[self.site_id].reshape(3, 3)
            mujoco.mju_mat2Quat(quat_err, (r_target @ r_now.T).flatten())
            mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
            err = np.concatenate([target_pos - self.data.site_xpos[self.site_id], w * rot_err])
            jac = np.vstack([self._jacp[:, self.dof_addr], w * self._jacr[:, self.dof_addr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * np.eye(6), err)
            q = np.clip(
                q + np.clip(dq, -self.max_step_rad, self.max_step_rad), self.q_low, self.q_high
            )
        pos, rot = self.tcp_pose(q)
        tilt = float(np.degrees(np.arccos(np.clip(-rot[2, 2], -1.0, 1.0))))
        return IKResult(
            q=q, pos_err_m=float(np.linalg.norm(target_pos - pos)), tilt_deg=tilt, yaw=yaw
        )

    def solve_best_yaw(
        self,
        target_pos: np.ndarray,
        yaws: list[float],
        q_seed: np.ndarray | None = None,
        roll_penalty: float = 0.01,
    ) -> IKResult:
        """候補の閉じ方向から、誤差が小さく手首ロールの回転量も小さい解を選ぶ.

        閉じ方向は向きのない軸として扱い、各候補 ``yaw`` について ``yaw + π`` も試す
        （指を左右入れ替えるだけで同じ挟み方になる）。直方体なら面に垂直な 2 軸
        （``obj_yaw`` と ``obj_yaw + π/2``）を渡す。手首ロールは関節範囲が約 ±160° なので、
        片方の向きだけだと限界に当たって解けないことがある。
        """
        results = [
            self.solve(target_pos, yaw + flip, q_seed) for yaw in yaws for flip in (0.0, np.pi)
        ]
        return min(results, key=lambda r: r.pos_err_m + roll_penalty * abs(r.q[4]))


def reachability_grid(
    ik: TopDownIK,
    xs: np.ndarray,
    ys: np.ndarray,
    z: float,
    yaws: tuple[float, ...] = (0.0, np.pi / 2),
    pos_tol_m: float = 0.003,
    tilt_tol_deg: float = 5.0,
) -> np.ndarray:
    """高さ ``z`` で、どの閉じ方向 ``yaws`` でも IK が解ける点を bool 配列で返す.

    戻り値の形は ``(len(ys), len(xs))``。小片の向きはランダムなので、
    全候補の向きで解けることを条件にする。姿勢を問わず位置だけ見たい場合は
    ``rot_weight`` を小さくした ``ik`` と ``tilt_tol_deg=180`` を渡す。
    """
    grid = np.zeros((len(ys), len(xs)), dtype=bool)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            target = np.array([x, y, z])
            grid[iy, ix] = all(
                ik.solve_best_yaw(target, [yaw]).ok(pos_tol_m, tilt_tol_deg) for yaw in yaws
            )
    return grid
