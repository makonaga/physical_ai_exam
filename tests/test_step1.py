"""段階1の部品（鍋投入タスク環境・エキスパート・データの単位変換）の動作確認. 描画は使わない."""

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from so101_food.datagen.schema import dataset_features, from_dataset_units, to_dataset_units  # noqa: E402
from so101_food.env.pot_task import PotSpec, PotTaskEnv  # noqa: E402
from so101_food.expert.pot_expert import PotExpert, run_expert_episode  # noqa: E402


@pytest.fixture(scope="module")
def env():
    env = PotTaskEnv()
    env.reset(seed=0)
    yield env
    env.close()


def test_pot_contains():
    pot = PotSpec(center=(0.2, 0.2), inner_radius=0.05, height=0.05)
    assert pot.contains(np.array([0.2, 0.2, 0.02]))
    assert not pot.contains(np.array([0.2, 0.2, 0.08]))  # 縁より上（まだ持っている）
    assert not pot.contains(np.array([0.27, 0.2, 0.01]))  # 鍋の外


def test_piece_dropped_into_pot_is_success(env):
    env.reset(seed=0)
    addr = env._slots[0].qpos_addr
    cx, cy = env.pot.center
    env.data.qpos[addr : addr + 3] = [cx, cy, 0.10]
    hold = env.robot_qpos().astype(np.float32)
    info = {}
    for _ in range(60):
        _, _, _, _, info = env.step(hold)
    assert info["in_pot"] and info["success"]


def test_expert_puts_piece_into_pot(env):
    expert = PotExpert(env)
    records = [run_expert_episode(env, expert, seed) for seed in range(5)]
    assert sum(r.success for r in records) >= 4, [r.reason for r in records]
    rec = records[0]
    assert rec.qpos.shape == (rec.n_frames, env.model.nq)
    assert rec.state.shape == rec.action.shape == (rec.n_frames, 6)


def test_dataset_units_roundtrip():
    q = np.array([0.1, -0.5, 0.7, 1.2, -0.3, 0.4], dtype=np.float32)
    assert np.allclose(from_dataset_units(to_dataset_units(q)), q, atol=1e-5)
    features = dataset_features()
    assert features["observation.state"]["shape"] == (6,)
    assert {"observation.images.overhead", "observation.images.wrist"} <= set(features)
