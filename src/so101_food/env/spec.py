"""タスクの設定（何を・どこに置くか）. MuJoCo を読み込まないので、描画方式を選ぶ前にも使える."""

from __future__ import annotations

from dataclasses import asdict, dataclass

TASKS = ("cube", "pieces")


@dataclass(frozen=True)
class TaskSpec:
    task: str = "cube"
    cube_mm: float = 25.0
    piece_set: str = "carrot_train"
    pool_size: int = 20
    spawn_radius: float = 0.07

    def to_dict(self) -> dict:
        return asdict(self)
