"""食材の小片を鍋に入れるタスクの環境.

so101-nexus の ``PickEnv``（小片の配置・把持判定）を土台に、固定の鍋と
俯瞰カメラをシーンに加え、成功判定を「小片が鍋の中にある」に差し替える。

- 制御は ``pd_joint_pos``（6 関節の目標角度 [rad]）。物理の制御周期は 50Hz で、
  ポリシーとデータセットは 25Hz（同じ目標を 2 ステップ保持）で動かす。
- カメラ画像は so101-nexus の観測ではなく :mod:`so101_food.env.cameras` で描画する
  （データ生成時は状態列から描画し直すため）。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from so101_nexus import CubeObject, MeshObject, PickConfig, RenderConfig
from so101_nexus.constants import sample_color
from so101_nexus.mujoco.pick_env import PickEnv
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


# 手首カメラの取り付け位置（gripper ボディ座標）・下向きの角度・画角。
# so101-nexus の WristCamera の既定ランダム化範囲の中央値（ランダム化はしない）
WRIST_CAM_POS = (0.0, 0.04, -0.04)
WRIST_CAM_PITCH_DEG = -17.2
WRIST_CAM_FOVY = 75.0


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


def default_pick_config(objects=None, spawn_radius: float = 0.07) -> PickConfig:
    """段階1の既定: 25mm 角の立方体 1 個を、ロボット前方の半径 7cm の円内に置く.

    描画品質は CPU 描画でも速い設定（影・アンチエイリアスなし）に固定する。
    学習データと評価で見た目を揃えるため、実行環境によって変えない。
    """
    return PickConfig(
        objects=objects if objects is not None else CubeObject(side_length_mm=25, color="orange"),
        spawn_center=(0.21, 0.0),
        spawn_min_radius=0.0,
        spawn_max_radius=spawn_radius,
        spawn_angle_half_range_deg=180.0,
        terminate_on_success=False,
        render=RenderConfig(quality=quality_preset("fast"), camera="side"),
    )


class PotTaskEnv(PickEnv):
    """小片を掴んで鍋に入れるタスク. ``info["success"]`` は小片が鍋の中にあるとき True."""

    def __init__(
        self,
        config: PickConfig | None = None,
        render_mode: str | None = None,
        robot_init_qpos_noise: float = 0.02,
        pot: PotSpec | None = None,
        overhead: OverheadCameraSpec | None = None,
    ) -> None:
        # so101-nexus 0.10.0 の PickEnv.__init__ と同じ手順に、鍋とカメラの追加だけを加えている
        config = config if config is not None else default_pick_config()
        self.pot = pot if pot is not None else PotSpec()
        self.overhead = overhead if overhead is not None else OverheadCameraSpec()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode="pd_joint_pos",
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        scene_objects = list(config.objects)
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

    @property
    def task_description(self) -> str:
        return TASK_DESCRIPTION

    def robot_qpos(self) -> np.ndarray:
        """6 関節（腕 5 + グリッパ）の現在角度 [rad]."""
        return self.data.qpos[self.robot_qpos_addr].copy()

    def object_pose(self) -> np.ndarray:
        """対象の小片の姿勢 ``[x, y, z, qw, qx, qy, qz]``."""
        return self._get_target_pose()

    def target_vertices(self) -> np.ndarray:
        """対象の小片の衝突形状の頂点（世界座標、(N, 3)）. 把持計画に使う."""
        verts = []
        for gid in self._slots[self._target_slot_idx].geom_ids:
            pos = self.data.geom_xpos[gid]
            rot = self.data.geom_xmat[gid].reshape(3, 3)
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

    def _task_reset(self) -> None:
        # so101-nexus は使わない片を同じ場所（地下）に重ねて置き、次の reset の途中で一時的に
        # 接触を有効に戻す。片が多いと重なった片どうしの接触が爆発的に増えて MuJoCo が
        # メモリ不足で止まるため、reset の前に待機中の片を上空に離して並べておく
        for i, slot in enumerate(self._slots):
            self.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = [10.0 + 0.5 * i, 0.0, 5.0]
        super()._task_reset()
        self._max_lift = 0.0
        self._ever_grasped = False
        target = self._slots[self._target_slot_idx]
        if isinstance(target.obj, MeshObject):
            # 食材片の表示用ジオムの色を、食材の基本色から少しずらして塗る
            food = target.obj.name.split()[0]
            rgb = np.array(FOOD_RGB.get(food, (0.7, 0.7, 0.7)))
            rgb = rgb * self.np_random.uniform(1 - COLOR_JITTER, 1 + COLOR_JITTER, size=3)
            visual = self.model.geom(f"pick_slot_{self._target_slot_idx}_visual").id
            self.model.geom_rgba[visual] = [*np.clip(rgb, 0, 1), 1.0]

    def _get_info(self) -> dict:
        info = super()._get_info()
        obj_pos = self._get_target_pose()[:3]
        self._max_lift = max(self._max_lift, info["lift_height"])
        self._ever_grasped = self._ever_grasped or info["is_grasped"] > 0.5
        info.update(
            obj_pos=obj_pos,
            in_pot=self.pot.contains(obj_pos),
            max_lift=self._max_lift,
            ever_grasped=self._ever_grasped,
        )
        info["success"] = info["in_pot"]
        return info

    def _compute_reward(self, info: dict) -> float:
        return float(info["success"])


def failure_reason(info: dict) -> str:
    """エピソード終了時の ``info`` から結果を分類する."""
    if info["success"]:
        return "success"
    if not info["ever_grasped"] or info["max_lift"] < 0.02:
        return "not_grasped"
    return "missed_pot"
