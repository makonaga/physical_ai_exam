"""ポリシーの入力画像（俯瞰カメラ・手首カメラ）を描画する.

環境の ``MjData`` とは別の ``MjData`` に関節角度（qpos）を写して描画するので、
シミュレーション中の状態にも、保存しておいた状態列の再描画にも同じ処理を使える。
データ生成と評価で同じクラスを使い、見た目を完全に揃える。
"""

from __future__ import annotations

import mujoco
import numpy as np

# データセットのキー名 → MJCF のカメラ名
CAMERAS = {"overhead": "overhead_cam", "wrist": "wrist_cam"}
IMAGE_SIZE = 224


class CameraRig:
    def __init__(self, model: mujoco.MjModel, size: int = IMAGE_SIZE, shadows: bool = False):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=size, width=size)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        self.camera_ids = {key: model.camera(name).id for key, name in CAMERAS.items()}

    def render(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """全関節の ``qpos``（``model.nq`` 次元）の状態で各カメラを描画する（HWC, uint8）."""
        self.data.qpos[:] = qpos
        # 描画に必要な位置（物体・カメラ・照明）だけ計算する（接触計算は不要）
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        images = {}
        for key, cam_id in self.camera_ids.items():
            self.renderer.update_scene(self.data, camera=cam_id)
            images[key] = self.renderer.render().copy()
        return images

    def close(self) -> None:
        self.renderer.close()
