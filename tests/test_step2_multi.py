"""段階2（複数の食材片から一番近い片を鍋に入れる）の動作確認. 描画は使わない."""

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from so101_food.env.factory import make_env  # noqa: E402
from so101_food.env.pot_task import READY_QPOS_DEG, ClutterSpec, _polygon_centroid  # noqa: E402
from so101_food.env.spec import TaskSpec  # noqa: E402
from so101_food.expert.pot_expert import PotExpert, run_expert_episode  # noqa: E402
from so101_food.shapes.cut import PieceSpec, support_margin  # noqa: E402
from so101_food.shapes.library import load_set  # noqa: E402


@pytest.fixture(scope="module")
def multi_env():
    env = make_env(
        TaskSpec(task="multi", piece_set="carrot_test", pool_size=24), np.random.default_rng(0)
    )
    env.scene_check = PotExpert(env).can_grasp
    yield env
    env.close()


def test_pieces_rest_stably():
    # 重心が接地面の端に近い片は、置いた直後に転がる。どの片も余裕をもって置ける
    spec = PieceSpec()
    for name in ("carrot_train", "carrot_test"):
        for piece in load_set(name):
            mesh = trimesh.load(piece.path, force="mesh")
            assert support_margin(mesh) >= spec.min_support_margin - 1e-6, piece.name


def _hull(xy: np.ndarray) -> np.ndarray:
    return xy[ConvexHull(xy).vertices]


def test_clutter_layout(multi_env):
    env, spec = multi_env, ClutterSpec()
    counts = set()
    for seed in range(10):
        _, info = env.reset(seed=seed)
        active = env.active_slots
        assert spec.n_pieces[0] <= info["n_pieces"] == len(active) <= spec.n_pieces[1]
        counts.add(len(active))
        # 対象（先頭）がロボットの根元に一番近く、2 番目との差が見て分かる程度にある
        dist = []
        for slot in active:
            pose = env.object_pose(slot)
            footprint = env._slot_vertices(slot)[:, :2]
            dist.append(np.linalg.norm(_polygon_centroid(_hull(footprint))))
            assert pose[2] < 0.03  # 机の上にある（待機場所ではない）
        assert np.argmin(dist) == 0
        assert sorted(dist)[1] - dist[0] >= spec.nearest_margin - 0.002
        # 片どうしは重なっていない（中心間が 1cm 以上）
        xy = np.array([env.object_pose(s)[:2] for s in active])
        gaps = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy))
        assert gaps.min() > 0.01
    assert len(counts) > 3  # 片の数がエピソードごとに変わる


def test_starts_from_ready_pose(multi_env):
    env = multi_env
    env.reset(seed=0)
    assert np.allclose(np.degrees(env.robot_qpos()), READY_QPOS_DEG, atol=2.0)
    assert env.task_description == "Put the food piece closest to the robot into the pot."


def test_expert_puts_nearest_piece_into_pot(multi_env):
    env = multi_env
    expert = PotExpert(env)
    records = [run_expert_episode(env, expert, seed) for seed in range(6)]
    assert sum(r.success for r in records) >= 5, [r.reason for r in records]
    for r in records:
        if r.success:
            assert r.info["others_in_pot"] == 0  # 入れたのは対象の片だけ
