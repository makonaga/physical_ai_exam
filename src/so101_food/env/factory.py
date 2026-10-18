"""タスクの設定（何を・どこに置くか）から環境を作る.

- ``cube``  : 段階1。25mm 角の立方体（大きさは ``cube_mm`` で変えられる）
- ``pieces``: 段階2。形状セットの食材片（形・大きさがランダム）

食材片は形状セットから ``pool_size`` 個を選んで環境に組み込み、エピソードごとに
その中から 1 個を置く。全部を 1 つの環境に入れるとモデルが大きくなりすぎるため、
データ生成・評価では一定エピソードごとに選び直して環境を作り直す。
"""

from __future__ import annotations

import numpy as np
from so101_nexus import CubeObject, MeshObject

from so101_food.env.pot_task import PotTaskEnv, default_pick_config
from so101_food.env.spec import TASKS, TaskSpec
from so101_food.shapes.library import load_set

__all__ = ["TASKS", "TaskSpec", "make_env"]


def make_env(spec: TaskSpec, rng: np.random.Generator) -> PotTaskEnv:
    if spec.task == "cube":
        objects = CubeObject(side_length_mm=spec.cube_mm, color="orange")
    elif spec.task == "pieces":
        pieces = load_set(spec.piece_set)
        chosen = rng.choice(len(pieces), size=min(spec.pool_size, len(pieces)), replace=False)
        objects = [
            MeshObject(
                collision_mesh_path=str(pieces[i].path),
                visual_mesh_path=str(pieces[i].path),
                mass=pieces[i].mass_kg,
                # 先頭の単語を食材名として色分けに使う（PotTaskEnv）
                name=f"{pieces[i].food} piece {pieces[i].name}",
            )
            for i in sorted(chosen)
        ]
    else:
        raise ValueError(f"unknown task: {spec.task} (choose from {TASKS})")
    return PotTaskEnv(config=default_pick_config(objects=objects, spawn_radius=spec.spawn_radius))
