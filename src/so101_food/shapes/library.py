"""食材片の形状セット（事前生成したメッシュの集まり）.

    python -m so101_food.shapes.library            # 既定のセットをすべて作る

形状セットは ``assets/pieces/<セット名>/`` に OBJ ファイルと ``index.json`` として置き、
リポジトリに含める（計算環境の違いで形が変わらないように、生成結果そのものを共有する）。
学習用（train）と評価用（test）は別の seed で作り、評価では学習で見せていない形を使える。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_food.shapes.cut import FOOD_STYLES, make_piece, piece_info

REPO = Path(__file__).resolve().parents[3]
PIECES_DIR = REPO / "assets" / "pieces"

# セット名 → (食材, 個数, seed)。切り方は食材ごとに決まる（shapes/cut.py の FOOD_STYLES）
DEFAULT_SETS = {
    "carrot_train": ("carrot", 300, 0),
    "carrot_test": ("carrot", 100, 1),
    "potato_train": ("potato", 300, 2),
    "potato_test": ("potato", 100, 3),
    "daikon_train": ("daikon", 300, 4),
    "daikon_test": ("daikon", 100, 5),
}


@dataclass(frozen=True)
class Piece:
    name: str
    food: str
    style: str
    mass_kg: float
    extents_mm: tuple[float, float, float]
    min_width_mm: float
    path: Path


def generate_set(name: str, food: str, count: int, seed: int, out_dir: Path = PIECES_DIR) -> Path:
    rng = np.random.default_rng(seed)
    set_dir = out_dir / name
    set_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    for i in range(count):
        styles = FOOD_STYLES[food]
        style = styles[i % len(styles)]
        mesh = make_piece(food, style, rng)
        piece_name = f"{food}_{i:04d}"
        mesh.export(set_dir / f"{piece_name}.obj")
        infos.append(piece_info(piece_name, food, style, mesh).to_dict())
    index = {"name": name, "food": food, "seed": seed, "pieces": infos}
    (set_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return set_dir


def load_set(name: str, pieces_dir: Path = PIECES_DIR) -> list[Piece]:
    """形状セットを読む. なければ既定の設定で作る."""
    set_dir = pieces_dir / name
    if not (set_dir / "index.json").exists():
        if name not in DEFAULT_SETS:
            raise FileNotFoundError(f"piece set not found: {set_dir}")
        generate_set(name, *DEFAULT_SETS[name], out_dir=pieces_dir)
    index = json.loads((set_dir / "index.json").read_text())
    return [
        Piece(
            name=p["name"],
            food=p["food"],
            style=p["style"],
            mass_kg=p["mass_kg"],
            extents_mm=tuple(p["extents_mm"]),
            min_width_mm=p["min_width_mm"],
            path=set_dir / f"{p['name']}.obj",
        )
        for p in index["pieces"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(DEFAULT_SETS), help="作るセット名")
    args = ap.parse_args()
    for name in args.sets:
        set_dir = generate_set(name, *DEFAULT_SETS[name])
        pieces = load_set(name)
        ext = np.array([p.extents_mm for p in pieces])
        print(
            f"{set_dir}: {len(pieces)} 個, 高さ {ext[:, 2].min():.0f}〜{ext[:, 2].max():.0f}mm, "
            f"上から見た寸法 {ext[:, :2].min():.0f}〜{ext[:, :2].max():.0f}mm"
        )


if __name__ == "__main__":
    main()
