"""タスクの設定（何を・どこに置くか）から環境を作る.

- ``cube``  : 段階1。25mm 角の立方体 1 個（大きさは ``cube_mm`` で変えられる）
- ``pieces``: 形・大きさがランダムな食材片 1 個
- ``multi`` : 段階2。食材片を ``min_pieces``〜``max_pieces`` 個散らばせ、ロボットに一番近い片を掴む
- ``lang``  : 段階3。``foods`` の食材の片を混ぜて散らばせ、指示文で指定した食材の片のうち
  ロボットに一番近い片を掴む。形状セットは ``<食材>_<piece_set の末尾（train / test）>`` を使う

食材片は形状セットから ``pool_size`` 個（``multi`` / ``lang`` では ``max_pieces`` 個以上）を選んで
環境に組み込み、エピソードごとにその中から置く片を選ぶ。全部を 1 つの環境に入れると
モデルが大きくなりすぎるため、データ生成・評価では一定エピソードごとに選び直して環境を作り直す。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from so101_nexus import CubeObject, MeshObject

from so101_food.env.pot_task import READY_QPOS_DEG, ClutterSpec, PotTaskEnv, default_pick_config
from so101_food.env.spec import TASKS, TaskSpec
from so101_food.shapes.library import load_set

__all__ = ["TASKS", "TaskSpec", "make_env"]


def _piece_objects(spec: TaskSpec, rng: np.random.Generator, count: int) -> list[MeshObject]:
    pieces = load_set(spec.piece_set)
    chosen = rng.choice(len(pieces), size=min(count, len(pieces)), replace=False)
    return [
        MeshObject(
            collision_mesh_path=str(pieces[i].path),
            visual_mesh_path=str(pieces[i].path),
            mass=pieces[i].mass_kg,
            # 先頭の単語を食材名として色分けに使う（PotTaskEnv）
            name=f"{pieces[i].food} piece {pieces[i].name}",
        )
        for i in sorted(chosen)
    ]


def split_of(piece_set: str) -> str:
    """形状セット名の末尾（train / test）. "carrot_test" → "test"、"test" → "test"."""
    return piece_set.rsplit("_", 1)[-1]


def make_env(spec: TaskSpec, rng: np.random.Generator) -> PotTaskEnv:
    clutter, rest = None, None
    if spec.task == "cube":
        objects = CubeObject(side_length_mm=spec.cube_mm, color="orange")
    elif spec.task == "pieces":
        objects = _piece_objects(spec, rng, spec.pool_size)
    elif spec.task == "multi":
        objects = _piece_objects(spec, rng, max(spec.pool_size, spec.max_pieces))
        clutter = ClutterSpec(n_pieces=(spec.min_pieces, spec.max_pieces))
        rest = READY_QPOS_DEG  # 作業範囲を隠さないよう、手先を上げた姿勢から始める
    elif spec.task == "lang":
        # 食材ごとに同じ数ずつ選んで混ぜる（例: 3 食材 × 8 個 = 24 個）
        per_food = math.ceil(max(spec.pool_size, spec.max_pieces) / len(spec.foods))
        split = split_of(spec.piece_set)
        objects = [
            obj
            for food in spec.foods
            for obj in _piece_objects(replace(spec, piece_set=f"{food}_{split}"), rng, per_food)
        ]
        clutter = ClutterSpec(n_pieces=(spec.min_pieces, spec.max_pieces), language=True)
        rest = READY_QPOS_DEG
    else:
        raise ValueError(f"unknown task: {spec.task} (choose from {TASKS})")
    return PotTaskEnv(
        config=default_pick_config(
            objects=objects, spawn_radius=spec.spawn_radius, rest_qpos_deg=rest
        ),
        clutter=clutter,
    )
