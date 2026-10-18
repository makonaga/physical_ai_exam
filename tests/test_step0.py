"""段階0の部品（グリッパ計測・真上把持 IK・把持テスト）の動作確認. 描画は使わない."""

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import so101_nexus.mujoco  # noqa: E402, F401
from so101_nexus import CubeObject, PickConfig  # noqa: E402

from so101_food.expert.ik import TopDownIK  # noqa: E402
from so101_food.expert.simple_pick import run_pick_trial  # noqa: E402
from so101_food.sim.gripper import angle_for_gap, measure_gripper  # noqa: E402


@pytest.fixture(scope="module")
def env():
    config = PickConfig(
        objects=CubeObject(side_length_mm=25),
        spawn_center=(0.20, 0.0),
        spawn_min_radius=0.0,
        spawn_max_radius=0.06,
        terminate_on_success=False,
    )
    env = gym.make("MuJoCoPickLift-v1", config=config, max_episode_steps=1000)
    env.reset(seed=0)
    yield env
    env.close()


def test_gripper_gap_increases_with_angle(env):
    samples = measure_gripper(env.unwrapped.model)
    gaps = [s.pad_gap_mm for s in samples]
    assert all(b > a for a, b in zip(gaps, gaps[1:], strict=False))
    # 2〜3cm 角の小片が入る開き角度が関節範囲内にある
    assert -10.0 < angle_for_gap(samples, 30.0) < 100.0


def test_top_down_ik_reaches_work_area(env):
    ik = TopDownIK(env.unwrapped.model)
    for target in ([0.20, 0.0, 0.0125], [0.15, 0.10, 0.0125], [0.25, -0.08, 0.05]):
        for yaw in (0.0, 0.7, 1.6, 2.5):
            assert ik.solve_best_yaw(np.array(target), [yaw]).ok(), (target, yaw)


def test_top_down_ik_rejects_out_of_reach(env):
    ik = TopDownIK(env.unwrapped.model)
    assert not ik.solve_best_yaw(np.array([0.45, 0.0, 0.0125]), [0.0]).ok()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_pick_lifts_cube(env, seed):
    ik = TopDownIK(env.unwrapped.model)
    result = run_pick_trial(env, ik, seed=seed)
    assert result["success"], result
