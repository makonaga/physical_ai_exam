"""食材の小片を鍋に入れるタスクの環境.

so101-nexus の ``PickEnv``（小片の配置・把持判定）を土台に、固定の鍋と
俯瞰カメラをシーンに加え、成功判定を「小片が鍋の中にある」に差し替える。

- 制御は ``pd_joint_pos``（6 関節の目標角度 [rad]）。物理の制御周期は 50Hz で、
  ポリシーとデータセットは 25Hz（同じ目標を 2 ステップ保持）で動かす。
- カメラ画像は so101-nexus の観測ではなく :mod:`so101_food.env.cameras` で描画する
  （データ生成時は状態列から描画し直すため）。
- ``clutter`` を指定すると、机の上に複数の食材片を散らばせ、ロボットに一番近い片を
  掴む対象にする（段階2）。指定しなければ机の上の物体は 1 個（段階1）。
- ``clutter.language`` を True にすると、エピソードごとに食材を 1 つ選んで指示文で伝え
  （例: "Put the potato piece closest to the robot into the pot."）、その食材の片のうち
  ロボットに一番近い片を対象にする（段階3）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from so101_nexus import CubeObject, MeshObject, PickConfig, RenderConfig
from so101_nexus.config import RobotConfig
from so101_nexus.constants import sample_color
from so101_nexus.mujoco.pick_env import PickEnv
from so101_nexus.mujoco.spawn_utils import align_freejoint_geom_to_floor, set_slot_contacts
from so101_nexus.object_slots import compile_object_scene
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

from so101_food.utils.render import quality_preset

ROBOT_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
PHYSICS_HZ = 50  # so101-nexus の制御周期（0.02 秒）
CONTROL_HZ = 25  # ポリシー・データセットの周期
FRAME_SKIP = PHYSICS_HZ // CONTROL_HZ
TASK_DESCRIPTION = "Put the food piece into the pot."
TASK_DESCRIPTION_CLUTTER = "Put the food piece closest to the robot into the pot."
# 段階3: 食材を言葉で指定する指示文（{food} に carrot / potato / daikon が入る）
TASK_DESCRIPTION_LANGUAGE = "Put the {food} piece closest to the robot into the pot."
POT_RGBA = "0.55 0.57 0.62 1"  # 金属っぽい灰色
# 食材片の基本色（RGB）。エピソードごとに明るさ・色味を少し揺らす
FOOD_RGB = {
    "carrot": (0.93, 0.45, 0.10),
    "potato": (0.80, 0.68, 0.40),
    "daikon": (0.93, 0.93, 0.88),
}
COLOR_JITTER = 0.08


@dataclass(frozen=True)
class PotSpec:
    """鍋（固定）の寸法 [m]. 側面は ``n_segments`` 枚の板で円筒を近似する."""

    center: tuple[float, float] = (0.20, 0.20)
    inner_radius: float = 0.055
    height: float = 0.05
    wall: float = 0.006
    bottom: float = 0.006
    n_segments: int = 20

    def contains(self, pos: np.ndarray, margin: float = 0.005) -> bool:
        """点 ``pos``（小片の中心）が鍋の内側（縁より下）にあるか."""
        r = np.hypot(pos[0] - self.center[0], pos[1] - self.center[1])
        return bool(r < self.inner_radius - margin and pos[2] < self.height)


@dataclass(frozen=True)
class OverheadCameraSpec:
    """作業領域と鍋を斜め上から映す固定カメラ（ロボットの正面側に置く）."""

    pos: tuple[float, float, float] = (0.40, 0.085, 0.46)
    target: tuple[float, float, float] = (0.18, 0.085, 0.0)
    fovy: float = 45.0


# 複数片のタスクの開始姿勢 [deg]（肩パン, 肩リフト, 肘, 手首ピッチ, 手首ロール, グリッパ）。
# so101-nexus の休止姿勢（肘 90°, 手首 37.8°）では手先が机から約 6cm の高さで作業範囲の
# 上にかぶさり、俯瞰カメラから根元に近い片（＝掴む対象）が隠れる。手先を約 14cm に上げて
# 真下近くに向けると、俯瞰カメラから作業範囲が隠れず、手首カメラにも作業範囲の約 9 割が写る
READY_QPOS_DEG = (0.0, -70.0, 35.0, 90.0, 0.0, -10.0)

# 手首カメラの取り付け位置（gripper ボディ座標）・下向きの角度・画角。
# so101-nexus の WristCamera の既定ランダム化範囲の中央値（ランダム化はしない）
WRIST_CAM_POS = (0.0, 0.04, -0.04)
WRIST_CAM_PITCH_DEG = -17.2
WRIST_CAM_FOVY = 75.0


@dataclass(frozen=True)
class ClutterSpec:
    """机の上に複数の食材片を散らばせる置き方 [m]. 位置は世界座標（ロボットの根元が原点）.

    範囲は「俯瞰カメラに写る」「鍋に重ならない」「ロボットの根元に近すぎない（真上から
    掴めるのは根元から 12〜30cm）」で決めている。片どうしは上から見た形（凸多角形）が
    ``gap`` 以上離れるように置く。
    """

    n_pieces: tuple[int, int] = (5, 20)  # 片の数（エピソードごとに一様乱数、両端を含む）
    x_range: tuple[float, float] = (0.11, 0.34)  # 片の全体がこの範囲に収まるように置く
    y_range: tuple[float, float] = (-0.085, 0.13)
    min_base_dist: float = 0.12  # ロボットの根元からの距離の下限
    pot_clearance: float = 0.01  # 鍋の外側との隙間
    gap: float = 0.003  # 片どうしの隙間の下限
    # 一番近い片と 2 番目に近い片の距離の差の下限（どれが一番近いか画像で分かるように）
    nearest_margin: float = 0.01
    max_target_dist: float = 0.27  # 一番近い片がこれより遠い配置はやり直す（腕が届かない）
    place_tries: int = 100  # 1 個の片を置く位置を探す回数
    scene_tries: int = 30  # 配置全体をやり直す回数の上限
    # True なら、机の上にある食材から 1 つを選んで指示文で伝え、その食材の片のうち一番近い片を
    # 対象にする（一番近い片・2 番目との差の条件も、その食材の片の中で判定する）
    language: bool = False


def pot_xml(pot: PotSpec) -> str:
    """鍋の MJCF（worldbody 直下に置く body）を返す."""
    r_wall = pot.inner_radius + pot.wall / 2
    half_chord = (pot.inner_radius + pot.wall) * np.tan(np.pi / pot.n_segments) * 1.05
    geoms = [
        f'<geom name="pot_bottom" type="cylinder" size="{pot.inner_radius + pot.wall:.4f} '
        f'{pot.bottom / 2:.4f}" pos="0 0 {pot.bottom / 2:.4f}" rgba="{POT_RGBA}"/>'
    ]
    for i in range(pot.n_segments):
        th = 2 * np.pi * i / pot.n_segments
        geoms.append(
            f'<geom name="pot_wall_{i}" type="box" size="{pot.wall / 2:.4f} {half_chord:.4f} '
            f'{pot.height / 2:.4f}" pos="{r_wall * np.cos(th):.4f} {r_wall * np.sin(th):.4f} '
            f'{pot.height / 2:.4f}" euler="0 0 {th:.4f}" rgba="{POT_RGBA}"/>'
        )
    cx, cy = pot.center
    body = "\n".join("      " + g for g in geoms)
    return f'    <body name="pot" pos="{cx} {cy} 0">\n{body}\n    </body>\n'


def lookat_camera_xml(name: str, cam: OverheadCameraSpec) -> str:
    """``pos`` から ``target`` を向く固定カメラの MJCF を返す."""
    pos, target = np.array(cam.pos), np.array(cam.target)
    forward = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xyaxes = " ".join(f"{v:.5f}" for v in (*right, *up))
    p = " ".join(f"{v:.4f}" for v in pos)
    return f'    <camera name="{name}" pos="{p}" xyaxes="{xyaxes}" fovy="{cam.fovy}"/>\n'


def default_pick_config(
    objects=None, spawn_radius: float = 0.07, rest_qpos_deg: tuple[float, ...] | None = None
) -> PickConfig:
    """段階1の既定: 25mm 角の立方体 1 個を、ロボット前方の半径 7cm の円内に置く.

    描画品質は CPU 描画でも速い設定（影・アンチエイリアスなし）に固定する。
    学習データと評価で見た目を揃えるため、実行環境によって変えない。
    ``rest_qpos_deg`` で開始姿勢を変えられる（省略時は so101-nexus の休止姿勢）。
    """
    robot = RobotConfig(rest_qpos_deg=rest_qpos_deg) if rest_qpos_deg is not None else None
    return PickConfig(
        robot=robot,
        objects=objects if objects is not None else CubeObject(side_length_mm=25, color="orange"),
        spawn_center=(0.21, 0.0),
        spawn_min_radius=0.0,
        spawn_max_radius=spawn_radius,
        spawn_angle_half_range_deg=180.0,
        terminate_on_success=False,
        render=RenderConfig(quality=quality_preset("fast"), camera="side"),
    )


def _rot2d(poly: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return poly @ np.array([[c, s], [-s, c]])


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """凸多角形（反時計回りの頂点列）の重心."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * area)


def _separated(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    """凸多角形 ``a`` と ``b`` が ``gap`` 以上離れているか（分離軸による保守的な判定）."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pa, pb = a @ normals.T, b @ normals.T
        if np.any(
            (pb.min(axis=0) - pa.max(axis=0) >= gap) | (pa.min(axis=0) - pb.max(axis=0) >= gap)
        ):
            return True
    return False


class PotTaskEnv(PickEnv):
    """小片を掴んで鍋に入れるタスク. ``info["success"]`` は対象の小片が鍋の中にあるとき True.

    ``clutter`` を指定すると、エピソードごとに ``clutter.n_pieces`` 個の片を机に置き、
    ロボットの根元に一番近い片を対象にする。``scene_check`` に関数を入れておくと、
    配置のたびに呼び、False なら配置をやり直す（データ生成・評価では「エキスパートが
    隣の片に指を当てずに掴める」配置だけを使うため）。
    """

    def __init__(
        self,
        config: PickConfig | None = None,
        render_mode: str | None = None,
        robot_init_qpos_noise: float = 0.02,
        pot: PotSpec | None = None,
        overhead: OverheadCameraSpec | None = None,
        clutter: ClutterSpec | None = None,
    ) -> None:
        # so101-nexus 0.10.0 の PickEnv.__init__ と同じ手順に、鍋とカメラの追加だけを加えている
        config = config if config is not None else default_pick_config()
        self.pot = pot if pot is not None else PotSpec()
        self.overhead = overhead if overhead is not None else OverheadCameraSpec()
        self.clutter = clutter
        self.scene_check: Callable[[PotTaskEnv], bool] | None = None
        # 指示する食材を決めておく（デモ用。None なら机の上の食材からランダムに選ぶ）
        self.forced_food: str | None = None
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode="pd_joint_pos",
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        scene_objects = list(config.objects)
        if clutter is not None and len(scene_objects) < clutter.n_pieces[1]:
            raise ValueError(
                f"{clutter.n_pieces[1]} 個置くには食材片が {len(scene_objects)} 個では足りません"
            )
        slot_names = [f"pick_slot_{i}" for i in range(len(scene_objects))]
        extra = pot_xml(self.pot) + lookat_camera_xml("overhead_cam", self.overhead)
        self.model, self._slots = compile_object_scene(
            scene_objects,
            slot_names,
            sample_color(config.ground_colors),
            option_xml=MUJOCO_SCENE_OPTION_XML,
            model_name="pot_scene",
            extra_bodies=extra,
        )
        self.data = mujoco.MjData(self.model)
        self._target_slot_idx = 0
        self._task_description = ""
        self._initial_obj_z = 0.0
        self._prev_task_potential = 0.0
        self._prev_reach_progress = 0.0
        self._prev_grasp_progress = 0.0
        self._set_target_geoms(self._slots[0].geom_ids)
        self._finish_model_setup()

        wrist = self.model.camera("wrist_cam").id
        self.model.cam_pos[wrist] = WRIST_CAM_POS
        mujoco.mju_euler2Quat(
            self.model.cam_quat[wrist], np.radians([WRIST_CAM_PITCH_DEG, 0.0, 0.0]), "XYZ"
        )
        self.model.cam_fovy[wrist] = WRIST_CAM_FOVY

        self.robot_qpos_addr = np.array([self.model.joint(n).qposadr[0] for n in ROBOT_JOINTS])
        self._max_lift = 0.0
        self._ever_grasped = False
        self._active: list[int] = [0]  # 机の上にある片のスロット番号（先頭が対象）
        self._initial_z: dict[int, float] = {}
        self._max_other_lift = 0.0
        # 上から見た形（置いたときの向きでの凸多角形、ボディ原点が基準）。配置に使う
        self._footprints = (
            [self._footprint(i) for i in range(len(self._slots))] if clutter is not None else []
        )

    def close(self) -> None:
        # scene_check にはエキスパートのメソッドを入れる（エキスパートもこの環境を参照する）。
        # 外さないと参照が循環し、古い環境（MuJoCo のモデルなど、数百 MB）がすぐには解放されず、
        # 環境を作り直すたびにメモリが増える（並列のデータ生成でメモリ不足になった）
        self.scene_check = None
        super().close()

    @property
    def task_description(self) -> str:
        if self.clutter is None:
            return TASK_DESCRIPTION
        return self._task_description or TASK_DESCRIPTION_CLUTTER

    @property
    def target_food(self) -> str:
        """対象の片の食材名（carrot / potato / daikon。立方体なら cube）."""
        return self._slot_food(self._target_slot_idx)

    def _slot_food(self, slot: int) -> str:
        obj = self._slots[slot].obj
        return obj.name.split()[0] if isinstance(obj, MeshObject) else "cube"

    @property
    def active_slots(self) -> list[int]:
        """机の上にある片のスロット番号（先頭が対象の片）."""
        return list(self._active)

    def robot_qpos(self) -> np.ndarray:
        """6 関節（腕 5 + グリッパ）の現在角度 [rad]."""
        return self.data.qpos[self.robot_qpos_addr].copy()

    def object_pose(self, slot: int | None = None) -> np.ndarray:
        """小片（省略時は対象の片）の姿勢 ``[x, y, z, qw, qx, qy, qz]``."""
        addr = self._slots[self._target_slot_idx if slot is None else slot].qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _slot_vertices(self, slot: int, world: bool = True) -> np.ndarray:
        """スロットの衝突形状の頂点（(N, 3)）. ``world=False`` ならボディ座標."""
        verts = []
        for gid in self._slots[slot].geom_ids:
            if world:
                pos = self.data.geom_xpos[gid]
                rot = self.data.geom_xmat[gid].reshape(3, 3)
            else:
                pos = self.model.geom_pos[gid]
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, self.model.geom_quat[gid])
                rot = rot.reshape(3, 3)
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                half = self.model.geom_size[gid]
                local = np.array(np.meshgrid(*[[-h, h] for h in half])).reshape(3, -1).T
            elif self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.model.geom_dataid[gid]
                adr, num = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                local = self.model.mesh_vert[adr : adr + num]
            else:
                raise NotImplementedError(f"geom type {self.model.geom_type[gid]}")
            verts.append(local @ rot.T + pos)
        return np.vstack(verts)

    def target_vertices(self) -> np.ndarray:
        """対象の小片の衝突形状の頂点（世界座標、(N, 3)）. 把持計画に使う."""
        return self._slot_vertices(self._target_slot_idx)

    def neighbor_geoms(self, radius: float = 0.09) -> list[int]:
        """対象の片の近く（中心間が ``radius`` 以内）にある他の片の衝突ジオム."""
        center = self.object_pose()[:2]
        geoms = []
        for i in self._active[1:]:
            if np.linalg.norm(self.object_pose(i)[:2] - center) < radius:
                geoms.extend(self._slots[i].geom_ids)
        return geoms

    def _footprint(self, slot: int) -> np.ndarray:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(self._slots[slot].rest_quat, dtype=float))
        xy = (self._slot_vertices(slot, world=False) @ rot.reshape(3, 3).T)[:, :2]
        return xy[ConvexHull(xy).vertices]

    def _park(self, slot: int) -> None:
        """使わない片の接触を切り、机から離れた場所に重ならないように置く."""
        set_slot_contacts(self.model, self._slots[slot], False)
        addr = self._slots[slot].qpos_addr
        self.data.qpos[addr : addr + 7] = [10.0 + 0.5 * slot, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0]

    def _fits(self, poly: np.ndarray) -> bool:
        c = self.clutter
        (x0, x1), (y0, y1) = c.x_range, c.y_range
        if poly[:, 0].min() < x0 or poly[:, 0].max() > x1:
            return False
        if poly[:, 1].min() < y0 or poly[:, 1].max() > y1:
            return False
        if np.linalg.norm(poly, axis=1).min() < c.min_base_dist:
            return False
        pot_r = self.pot.inner_radius + self.pot.wall + c.pot_clearance
        return bool(np.linalg.norm(poly - np.asarray(self.pot.center), axis=1).min() >= pot_r)

    def _sample_layout(
        self, rng: np.random.Generator, n: int
    ) -> list[tuple[int, np.ndarray, float]] | None:
        """``n`` 個の片の種類・位置・向きを決める. 先頭が対象、残りは根元に近い順に並べて返す."""
        c = self.clutter
        chosen = rng.choice(len(self._slots), size=n, replace=False)
        placed: list[tuple[int, np.ndarray, float, np.ndarray]] = []
        lo, hi = np.array([c.x_range[0], c.y_range[0]]), np.array([c.x_range[1], c.y_range[1]])
        for slot in chosen:
            for _ in range(c.place_tries):
                xy, yaw = rng.uniform(lo, hi), rng.uniform(0.0, 2 * np.pi)
                poly = _rot2d(self._footprints[slot], yaw) + xy
                if self._fits(poly) and all(_separated(poly, q[3], c.gap) for q in placed):
                    placed.append((int(slot), xy, yaw, poly))
                    break
        dist = np.array([np.linalg.norm(_polygon_centroid(q[3])) for q in placed])
        order = list(np.argsort(dist))
        if len(placed) < n:
            return None
        candidates = order
        if c.language:
            # 机の上にある食材から 1 つ選び、その食材の片だけで一番近い片を決める
            foods = sorted({self._slot_food(q[0]) for q in placed})
            food = foods[int(rng.integers(len(foods)))]
            if self.forced_food is not None:
                # 乱数の使い方は変えずに食材だけ差し替える（同じ seed なら配置が同じになりやすい）
                if self.forced_food not in foods:
                    return None
                food = self.forced_food
            candidates = [i for i in order if self._slot_food(placed[i][0]) == food]
        if dist[candidates[0]] > c.max_target_dist:
            return None
        if len(candidates) > 1 and dist[candidates[1]] - dist[candidates[0]] < c.nearest_margin:
            return None
        target = candidates[0]
        return [placed[target][:3]] + [placed[i][:3] for i in order if i != target]

    def _place_layout(self, layout: list[tuple[int, np.ndarray, float]]) -> None:
        for slot, xy, yaw in layout:
            s = self._slots[slot]
            set_slot_contacts(self.model, s, True)
            quat = np.zeros(4)
            yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            mujoco.mju_mulQuat(quat, yaw_quat, np.asarray(s.rest_quat, dtype=float))
            align_freejoint_geom_to_floor(
                self.model,
                self.data,
                qpos_addr=s.qpos_addr,
                geom_ids=s.geom_ids,
                xy=(float(xy[0]), float(xy[1])),
                quat=quat,
            )

    def _clutter_reset(self) -> None:
        rng = self.np_random
        for slot in range(len(self._slots)):
            self._park(slot)
        # 片の数は先に決めて固定する（配置のやり直しで数が少ない方に偏らないように）
        lo, hi = self.clutter.n_pieces
        n = int(rng.integers(lo, hi + 1))
        layout = None
        for attempt in range(self.clutter.scene_tries):
            candidate = self._sample_layout(rng, n)
            if candidate is None:
                # 何度やっても置けないときは 1 個減らす（20 個でもほぼ起きない）
                if attempt % 10 == 9 and layout is None:
                    n = max(lo, n - 1)
                continue
            if layout is not None:
                for slot, _, _ in layout:
                    self._park(slot)
            layout = candidate
            self._place_layout(layout)
            self._active = [slot for slot, _, _ in layout]
            self._target_slot_idx = self._active[0]
            self._set_target_geoms(self._slots[self._target_slot_idx].geom_ids)
            if self.scene_check is None or self.scene_check(self):
                break
        if layout is None:
            raise RuntimeError("食材片を置ける配置が見つかりませんでした")
        if self.clutter.language:
            self._task_description = TASK_DESCRIPTION_LANGUAGE.format(food=self.target_food)
        else:
            self._task_description = TASK_DESCRIPTION_CLUTTER

    def _task_reset(self) -> None:
        if self.clutter is not None:
            self._clutter_reset()
        else:
            # so101-nexus は使わない片を同じ場所（地下）に重ねて置き、次の reset の途中で一時的に
            # 接触を有効に戻す。片が多いと重なった片どうしの接触が爆発的に増えて MuJoCo が
            # メモリ不足で止まるため、reset の前に待機中の片を上空に離して並べておく
            for i, slot in enumerate(self._slots):
                self.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = [10.0 + 0.5 * i, 0.0, 5.0]
            super()._task_reset()
            self._active = [self._target_slot_idx]
        self._max_lift = 0.0
        self._ever_grasped = False
        self._max_other_lift = 0.0
        self._moved_other_foods: set[str] = set()
        for slot in self._active:
            obj = self._slots[slot].obj
            if isinstance(obj, MeshObject):
                # 食材片の表示用ジオムの色を、食材の基本色から少しずらして塗る
                food = obj.name.split()[0]
                rgb = np.array(FOOD_RGB.get(food, (0.7, 0.7, 0.7)))
                rgb = rgb * self.np_random.uniform(1 - COLOR_JITTER, 1 + COLOR_JITTER, size=3)
                visual = self.model.geom(f"pick_slot_{slot}_visual").id
                self.model.geom_rgba[visual] = [*np.clip(rgb, 0, 1), 1.0]

    def _refresh_reset_reference_state(self) -> None:
        super()._refresh_reset_reference_state()
        self._initial_z = {i: float(self.object_pose(i)[2]) for i in self._active}

    def _get_info(self) -> dict:
        info = super()._get_info()
        obj_pos = self._get_target_pose()[:3]
        self._max_lift = max(self._max_lift, info["lift_height"])
        self._ever_grasped = self._ever_grasped or info["is_grasped"] > 0.5
        others_in_pot = 0
        for i in self._active[1:]:
            pos = self.object_pose(i)[:3]
            in_pot = self.pot.contains(pos)
            lift = pos[2] - self._initial_z.get(i, pos[2])
            others_in_pot += in_pot
            self._max_other_lift = max(self._max_other_lift, lift)
            if in_pot or lift >= 0.02:
                self._moved_other_foods.add(self._slot_food(i))
        target_food = self.target_food
        info.update(
            obj_pos=obj_pos,
            in_pot=self.pot.contains(obj_pos),
            max_lift=self._max_lift,
            ever_grasped=self._ever_grasped,
            n_pieces=len(self._active),
            target_food=target_food,
            others_in_pot=others_in_pot,
            max_other_lift=self._max_other_lift,
            # 対象と違う食材の片を持ち上げた・鍋に入れた（言葉の指示を取り違えた）
            moved_other_food=bool(self._moved_other_foods - {target_food}),
        )
        info["success"] = info["in_pot"]
        return info

    def _compute_reward(self, info: dict) -> float:
        return float(info["success"])


def failure_reason(info: dict) -> str:
    """エピソード終了時の ``info`` から結果を分類する.

    - ``success``     : 対象の片が鍋の中に入った
    - ``wrong_food``  : 対象の片は持ち上げず、指示と違う食材の片を鍋に入れた・持ち上げた（段階3）
    - ``wrong_piece`` : 対象の片は持ち上げず、同じ食材の別の片を鍋に入れた・持ち上げた
    - ``not_grasped`` : 対象の片を持ち上げられなかった
    - ``missed_pot``  : 対象の片を持ち上げたが鍋に入らなかった
    """
    if info["success"]:
        return "success"
    lifted = info["ever_grasped"] and info["max_lift"] >= 0.02
    if not lifted and info.get("moved_other_food", False):
        return "wrong_food"
    if not lifted and (info.get("others_in_pot", 0) > 0 or info.get("max_other_lift", 0) >= 0.02):
        return "wrong_piece"
    if not lifted:
        return "not_grasped"
    return "missed_pot"
