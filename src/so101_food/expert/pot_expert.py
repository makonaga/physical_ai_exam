"""小片を掴んで鍋に入れるスクリプテッド・エキスパート.

シミュレータから小片の真の形状（頂点）と姿勢を読み、関節空間の経由点を IK で求めて補間する。

1. 上空へ移動しながら指を開く（真上把持。上から見て一番細い方向で挟む。
   周りに他の片があるときは、指が他の片に当たらない方向・開き幅を選ぶ。
   手先が自分の腕に当たらないかも調べる）
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

import mujoco
import numpy as np

from so101_food.env.pot_task import CONTROL_HZ, FRAME_SKIP, PotTaskEnv, failure_reason
from so101_food.expert.grasp_plan import GraspTarget, grasp_candidates, plan_top_down_grasp
from so101_food.expert.ik import IKResult, TopDownIK
from so101_food.expert.trajectory import Segment, iter_actions
from so101_food.sim.gripper import angle_for_gap, measure_gripper

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
    # 離すときは大きく開く。小片が指に引っかかったまま離れないことがあるため（パッド間 約 87mm）
    release_deg: float = 70.0
    max_grasp_width: float = 0.045  # これより幅の広い向きしかない小片は掴まない
    close_deg: float = -10.0  # 閉じる目標の下限（完全に閉じた角度）
    # 閉じる目標は「パッド間が小片の幅よりこれだけ狭い角度」にする。目標と実際の差が握る力に
    # なるので、完全に閉じる目標だと力が強すぎ、小片が指先の小さな当たり判定に食い込んで
    # 離せなくなる。None なら常に close_deg まで閉じる
    squeeze_mm: float | None = 10.0
    approach_height: tuple[float, float] = (0.05, 0.07)  # 小片の上空の高さ（一様乱数の範囲）
    lift_height: float = 0.07  # 持ち上げ後の TCP の高さ（小片の中心から）
    carry_clearance: tuple[float, float] = (0.06, 0.08)  # 鍋の縁から TCP までの高さ
    drop_xy_noise: float = 0.01  # 落とす位置のばらつき（鍋の中心から）
    time_jitter: float = 0.15  # 区間の長さを ±15% 揺らす
    durations: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DURATIONS))
    # 周りに他の片があるとき: 開き幅を「小片の幅 + 両側にこの余裕」にする（open_deg が上限）
    open_margin_mm: float = 10.0
    # 上空へ移る・下降する間の、手先と他の片・自分の腕との距離の下限
    clearance: float = 0.004


# 他の片との接触を調べるロボットの部位（手首から先）
_HAND_BODIES = ("wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")
# 手先（手首より先）が自分の腕に当たらないかも調べる。根元に近い片を手首を大きくひねって
# 掴もうとすると、手首カメラの台座が肩に当たって下りられないことがある
_SELF_PAIRS = (
    ("gripper", "moving_jaw_so101_v1", "camera_mount"),
    ("shoulder", "upper_arm", "lower_arm"),
)


@dataclass(frozen=True)
class GraspChoice:
    target: GraspTarget
    grasp: IKResult  # 把持姿勢
    pre: IKResult  # 把持点の上空の姿勢
    g_open: float  # 下降するときのグリッパ角度 [rad]


@dataclass
class Plan:
    segments: list[Segment]
    grasp_yaw: float
    ik_ok: bool


def _collision_geoms(model: mujoco.MjModel, bodies: tuple[str, ...]) -> list[int]:
    ids = {model.body(name).id for name in bodies}
    return [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in ids and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


class PotExpert:
    def __init__(self, env: PotTaskEnv, params: ExpertParams | None = None) -> None:
        self.env = env
        self.params = params if params is not None else ExpertParams()
        self.grasp_ik = TopDownIK(env.model)
        # 鍋の上では位置を優先し、姿勢は「なるべく下向き」程度にする
        self.carry_ik = TopDownIK(env.model, rot_weight=0.02)
        self.gripper = measure_gripper(env.model)
        model = env.model
        self._check = mujoco.MjData(model)  # 指と他の片の距離を調べる作業用
        self._hand_geoms = _collision_geoms(model, _HAND_BODIES)
        self._self_pairs = [
            (g, h)
            for g in _collision_geoms(model, _SELF_PAIRS[0])
            for h in _collision_geoms(model, _SELF_PAIRS[1])
        ]
        self._arm_addr = env.robot_qpos_addr[:5]
        self._gripper_addr = env.robot_qpos_addr[5]
        angles = [s.angle_deg for s in self.gripper]
        gaps = [s.pad_gap_mm for s in self.gripper]
        self._max_open_gap_mm = float(np.interp(self.params.open_deg, angles, gaps))

    def _open_angle(self, width: float) -> float:
        gap = min(width * 1e3 + 2 * self.params.open_margin_mm, self._max_open_gap_mm)
        return float(np.radians(angle_for_gap(self.gripper, gap)))

    def _set_pose(self, q_arm: np.ndarray, gripper: float) -> None:
        self._check.qpos[self._arm_addr] = q_arm
        self._check.qpos[self._gripper_addr] = gripper
        mujoco.mj_kinematics(self.env.model, self._check)

    def _min_distance(self, pairs) -> float:
        model, data = self.env.model, self._check
        return min(
            (mujoco.mj_geomDistance(model, data, g, h, 0.02, None) for g, h in pairs),
            default=np.inf,
        )

    def _path_is_clear(self, choice: GraspChoice, neighbors: list[int]) -> bool:
        """今の姿勢 → 上空 → 把持姿勢と動く間、手先が他の片・自分の腕から十分離れているか.

        経路は関節空間の直線補間（実際の動きと同じ）で、途中の数点を調べる。
        """
        clearance = self.params.clearance
        self._check.qpos[:] = self.env.data.qpos
        near = [(g, h) for g in self._hand_geoms for h in neighbors]
        q_now = self.env.robot_qpos()
        q_pre, q_grasp = choice.pre.q, choice.grasp.q
        poses = [(q_pre + f * (q_grasp - q_pre), choice.g_open) for f in (1.0, 0.75, 0.5, 0.25)]
        poses += [
            (q_now[:5] + f * (q_pre - q_now[:5]), q_now[5] + f * (choice.g_open - q_now[5]))
            for f in (1.0, 0.8, 0.6, 0.4, 0.2)
        ]
        # 把持姿勢（一番低く、当たりやすい）から順に調べる
        for q_arm, gripper in poses:
            self._set_pose(q_arm, gripper)
            if self._min_distance(self._self_pairs) < clearance:
                return False
            if self._min_distance(near) < clearance:
                return False
        if not neighbors:
            return True
        # 開いた指が対象の片自体の上に乗っていないか（幅が開き幅ぎりぎりの向きなど）
        target_geoms = self.env._slots[self.env._target_slot_idx].geom_ids
        self._set_pose(q_grasp, choice.g_open)
        return self._min_distance([(g, h) for g in self._hand_geoms for h in target_geoms]) > 0.0

    def _grasp_poses(self, at: np.ndarray, yaws) -> list[IKResult]:
        """閉じる方向の候補（左右の入れ替え ``+π`` も含む）で解いた把持姿勢を、良い順に返す."""
        results = [self.grasp_ik.solve(at, yaw + flip) for yaw in yaws for flip in (0.0, np.pi)]
        ok = [r for r in results if r.ok()]
        return sorted(ok, key=lambda r: r.pos_err_m + 0.01 * abs(r.q[4]))

    def choose_grasp(self, approach_height: float) -> GraspChoice | None:
        """対象の片の掴み方（閉じる方向・把持点・上空の姿勢・開き幅）を選ぶ. なければ None."""
        p, env = self.params, self.env
        vertices = env.target_vertices()
        neighbors = env.neighbor_geoms()
        if neighbors:
            # 一番細い方向から順に、指が他の片に当たらない方向を探す
            candidates = [
                (t, (t.yaw,)) for t in grasp_candidates(vertices, max_width=p.max_grasp_width)
            ]
        else:
            # 周りに何もなければ、上から見て一番細い方向で挟む（形・大きさによらない）
            target = plan_top_down_grasp(vertices)
            if target.width >= p.max_grasp_width:
                return None
            candidates = [(target, target.yaw_candidates)]
        for target, yaws in candidates:
            at = target.center
            g_open = self._open_angle(target.width) if neighbors else float(np.radians(p.open_deg))
            for grasp in self._grasp_poses(at, yaws):
                # 真上姿勢がとれる高さはベースからの距離で変わるので、解けなければ上空の高さを下げる
                for height in (approach_height, 0.05, 0.04, 0.03):
                    pre = self.grasp_ik.solve(at + [0.0, 0.0, height], grasp.yaw, grasp.q)
                    if pre.ok():
                        break
                if not pre.ok():
                    continue
                choice = GraspChoice(target, grasp, pre, g_open)
                if self._path_is_clear(choice, neighbors):
                    return choice
        return None

    def can_grasp(self, env: PotTaskEnv | None = None) -> bool:
        """今の配置で対象の片を掴めるか（``PotTaskEnv.scene_check`` に渡す）."""
        return self.choose_grasp(self.params.approach_height[0]) is not None

    def plan(self, rng: np.random.Generator) -> Plan:
        """現在の状態（reset 直後）から、エピソード全体の区間列を作る."""
        p, env = self.params, self.env
        q0 = env.robot_qpos()
        choice = self.choose_grasp(rng.uniform(*p.approach_height))
        if choice is None:
            return Plan(segments=[], grasp_yaw=float("nan"), ik_ok=False)
        target, grasp, pre = choice.target, choice.grasp, choice.pre
        at = target.center
        lift = self.grasp_ik.solve(at + [0.0, 0.0, p.lift_height], grasp.yaw, grasp.q)
        cx, cy = env.pot.center
        drop = np.array(
            [
                cx + rng.uniform(-p.drop_xy_noise, p.drop_xy_noise),
                cy + rng.uniform(-p.drop_xy_noise, p.drop_xy_noise),
                env.pot.height + rng.uniform(*p.carry_clearance),
            ]
        )
        carry = self.carry_ik.solve(drop, grasp.yaw, lift.q)
        ik_ok = carry.pos_err_m < 0.005

        g_open, g_close = choice.g_open, np.radians(p.close_deg)
        if p.squeeze_mm is not None:
            gap = max(target.width * 1e3 - p.squeeze_mm, self.gripper[0].pad_gap_mm)
            g_close = max(g_close, np.radians(angle_for_gap(self.gripper, gap)))
        g_release = np.radians(p.release_deg)
        targets = [
            ("approach", pre.q, g_open),
            ("descend", grasp.q, g_open),
            ("close", grasp.q, g_close),
            ("lift", lift.q, g_close),
            ("carry", carry.q, g_close),
            ("release", carry.q, g_release),
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
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obj_pose0 = env.object_pose()
    plan = expert.plan(rng)
    qpos, state, action = [], [], []
    for _, act in iter_actions(plan.segments, env.robot_qpos(), ease=True):
        qpos.append(env.data.qpos.copy())
        state.append(env.robot_qpos())
        action.append(act)
        for _ in range(FRAME_SKIP):
            _, _, _, _, info = env.step(act)
    if not plan.segments:
        reason = "no_grasp"  # 指を他の片に当てずに掴める向きがない（実行しない）
    elif not plan.ik_ok:
        reason = "ik_failed"
    else:
        reason = failure_reason(info)
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
