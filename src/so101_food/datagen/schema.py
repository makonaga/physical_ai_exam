"""データセットの形式（LeRobot v3）と単位の変換.

関節角度は LeRobot の SO-101 実機（SO follower）と同じ単位で保存する:
腕の 5 関節は度、グリッパは関節範囲を 0〜100 に正規化した値。
将来の実機データと混ぜられるようにするため。
"""

from __future__ import annotations

import numpy as np
from so101_nexus import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

from so101_food.env.cameras import CAMERAS, IMAGE_SIZE
from so101_food.env.pot_task import ROBOT_JOINTS

ROBOT_TYPE = "so101_follower"
JOINT_NAMES = [f"{name}.pos" for name in ROBOT_JOINTS]


def dataset_features() -> dict:
    image = {
        "dtype": "video",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    }
    for key in CAMERAS:
        features[f"observation.images.{key}"] = dict(image)
    return features


def to_dataset_units(q_rad: np.ndarray) -> np.ndarray:
    """シミュレータの関節角度 [rad]（..., 6）→ データセットの単位."""
    return np.asarray(sim_qpos_to_dataset_row(np.asarray(q_rad)), dtype=np.float32)


def from_dataset_units(row: np.ndarray) -> np.ndarray:
    """データセットの単位（..., 6）→ シミュレータの関節角度 [rad]."""
    return np.asarray(dataset_row_to_sim_qpos(np.asarray(row)), dtype=np.float32)
