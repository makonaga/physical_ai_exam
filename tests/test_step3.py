"""段階3（指示文で指定した食材の一番近い片を鍋に入れる）の動作確認. 描画は使わない."""

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import trimesh  # noqa: E402

from so101_food.env.factory import make_env, split_of  # noqa: E402
from so101_food.env.pot_task import TASK_DESCRIPTION_LANGUAGE, failure_reason  # noqa: E402
from so101_food.env.spec import TaskSpec  # noqa: E402
from so101_food.expert.pot_expert import PotExpert, run_expert_episode  # noqa: E402
from so101_food.shapes.cut import FOOD_DENSITY, PieceSpec, support_margin  # noqa: E402
from so101_food.shapes.library import DEFAULT_SETS, load_set  # noqa: E402

FOODS = ("carrot", "potato", "daikon")


@pytest.fixture(scope="module")
def lang_env():
    env = make_env(TaskSpec(task="lang", piece_set="test", pool_size=24), np.random.default_rng(0))
    env.scene_check = PotExpert(env).can_grasp
    yield env
    env.close()


def test_all_piece_sets_are_committed_and_stable():
    spec = PieceSpec()
    for name, (food, count, _) in DEFAULT_SETS.items():
        pieces = load_set(name)
        assert len(pieces) == count
        assert {p.food for p in pieces} == {food}
        for piece in pieces[:20]:  # 全部は test_step2_multi と同じ考え方で時間がかかるので一部
            mesh = trimesh.load(piece.path, force="mesh")
            assert support_margin(mesh) >= spec.min_support_margin - 1e-6, piece.name


def test_food_shapes_and_densities():
    """じゃがいもは丸みのある乱切り、大根はいちょう切り。重さは食材ごとの密度 × 体積."""
    for food, style in (("potato", "rounded"), ("daikon", "ichou")):
        assert {p.style for p in load_set(f"{food}_train")} == {style}
    for food, density in FOOD_DENSITY.items():
        piece = load_set(f"{food}_test")[0]
        volume = trimesh.load(piece.path, force="mesh").volume
        assert piece.mass_kg == pytest.approx(volume * density, rel=1e-3)
    assert FOOD_DENSITY["potato"] > FOOD_DENSITY["carrot"] > FOOD_DENSITY["daikon"]


def test_split_of():
    assert split_of("carrot_train") == "train"
    assert split_of("carrot_test") == "test"
    assert split_of("test") == "test"


def test_language_target_is_nearest_piece_of_named_food(lang_env):
    env = lang_env
    named = set()
    assert {env._slot_food(i) for i in range(len(env._slots))} == set(FOODS)
    for seed in range(10):
        _, info = env.reset(seed=seed)
        food = env.target_food
        named.add(food)
        assert env.task_description == TASK_DESCRIPTION_LANGUAGE.format(food=food)
        assert info["target_food"] == food
        active = env.active_slots
        same = [s for s in active if env._slot_food(s) == food]
        dist = {s: np.linalg.norm(env.object_pose(s)[:2]) for s in same}
        assert min(dist, key=dist.get) == active[0]  # 対象は、指定した食材の中で一番近い片
    assert len(named) > 1  # 指定する食材がエピソードごとに変わる


def test_expert_follows_language(lang_env):
    env = lang_env
    expert = PotExpert(env)
    records = [run_expert_episode(env, expert, seed) for seed in range(5)]
    assert sum(r.success for r in records) >= 4, [r.reason for r in records]


def test_failure_reason_distinguishes_wrong_food():
    base = {"success": False, "ever_grasped": False, "max_lift": 0.0, "others_in_pot": 0}
    assert (
        failure_reason({**base, "max_other_lift": 0.05, "moved_other_food": True}) == "wrong_food"
    )
    assert (
        failure_reason({**base, "max_other_lift": 0.05, "moved_other_food": False}) == "wrong_piece"
    )
    assert failure_reason({**base, "max_other_lift": 0.0, "moved_other_food": False}) == (
        "not_grasped"
    )
