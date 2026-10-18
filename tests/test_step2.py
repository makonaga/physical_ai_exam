"""段階2の部品（形状生成・把持計画・食材片での実行）の動作確認. 描画は使わない."""

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import trimesh  # noqa: E402

from so101_food.env.factory import make_env  # noqa: E402
from so101_food.env.spec import TaskSpec  # noqa: E402
from so101_food.expert.grasp_plan import plan_top_down_grasp  # noqa: E402
from so101_food.expert.pot_expert import PotExpert, run_expert_episode  # noqa: E402
from so101_food.shapes.cut import CUT_STYLES, FOOD_TYPES, PieceSpec, make_piece, min_width  # noqa: E402
from so101_food.shapes.library import load_set  # noqa: E402


@pytest.mark.parametrize("food", FOOD_TYPES)
@pytest.mark.parametrize("style", CUT_STYLES)
def test_piece_meets_size_constraints(food, style):
    spec = PieceSpec()
    mesh = make_piece(food, style, np.random.default_rng(0), spec)
    assert mesh.is_volume
    assert min_width(mesh.vertices) >= spec.min_width
    assert mesh.extents.max() <= spec.max_extent + 1e-9
    # 置いたときの高さ（z）が外接箱で一番薄い（so101-nexus がそのまま置ける向き）
    assert mesh.extents[2] <= mesh.extents[:2].min() + 1e-6


def test_grasp_plan_box_closes_across_narrow_side():
    box = trimesh.creation.box((0.02, 0.03, 0.025))  # 幅 20mm の向きで挟むべき
    yaw = 0.3
    box.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
    box.apply_translation([0.2, 0.05, 0.0125])
    target = plan_top_down_grasp(box.vertices)
    assert target.width == pytest.approx(0.02, abs=1e-6)
    # 閉じる方向は箱の x 軸（yaw の向き）。軸の向きは問わない
    assert abs(np.sin(target.yaw - yaw)) < 1e-6
    assert np.allclose(target.center, [0.2, 0.05, 0.0125], atol=1e-6)
    assert len(target.yaw_candidates) == 1  # 直交方向（30mm）は幅が違うので候補にしない


def test_piece_sets_are_committed():
    for name, count in (("carrot_train", 300), ("carrot_test", 100)):
        pieces = load_set(name)
        assert len(pieces) == count
        assert all(p.path.exists() for p in pieces)


def test_expert_puts_random_pieces_into_pot():
    env = make_env(
        TaskSpec(task="pieces", piece_set="carrot_test", pool_size=5), np.random.default_rng(0)
    )
    expert = PotExpert(env)
    records = [run_expert_episode(env, expert, seed) for seed in range(5)]
    assert sum(r.success for r in records) >= 4, [r.reason for r in records]
    env.close()
