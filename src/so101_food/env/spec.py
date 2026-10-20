"""タスクの設定（何を・どこに置くか）. MuJoCo を読み込まないので、描画方式を選ぶ前にも使える."""

from __future__ import annotations

from dataclasses import asdict, dataclass

TASKS = ("cube", "pieces", "multi", "lang")
# 机に置く片の数の既定値（両端を含む）。段階3（lang）は段階2（multi）の 3 割増し
PIECE_COUNTS = {"multi": (5, 20), "lang": (7, 26)}


def piece_counts(task: str, lo: int | None = None, hi: int | None = None) -> tuple[int, int]:
    """机に置く片の数の範囲. 指定がなければタスクごとの既定値（multi / lang 以外は使わない）."""
    default_lo, default_hi = PIECE_COUNTS.get(task, (5, 20))
    return (default_lo if lo is None else lo, default_hi if hi is None else hi)


@dataclass(frozen=True)
class TaskSpec:
    task: str = "cube"
    cube_mm: float = 25.0
    piece_set: str = "carrot_train"
    pool_size: int = 20
    spawn_radius: float = 0.07
    # multi / lang で机に置く片の数（両端を含む。データ生成・評価では piece_counts で決める）
    min_pieces: int = 5
    max_pieces: int = 20
    foods: tuple[str, ...] = ("carrot", "potato", "daikon")  # lang で混ぜる食材

    def to_dict(self) -> dict:
        return asdict(self)
