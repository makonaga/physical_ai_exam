"""評価結果をまとめて比較する（成功率・失敗の内訳を、掴む片までの距離・片の数ごとに）.

    python -m so101_food.eval.analyze act_pot_multi_v1 act_pot_multi_v1_seen --dataset pot_multi_v1

``outputs/eval/<名前>/episodes.csv`` を読み、掴む片（評価開始時の位置）からロボットの根元までの
距離ごと・机の上の片の数ごとに成功率と失敗の内訳を表にする。``--dataset`` を付けると、
学習データで掴んだ片までの距離の分布も並べる（学習データに少ない位置で失敗していないかを見る）。

評価に手先の着地位置の記録（``land_*``）があれば、区間ごとに「着地のずれ」（水平距離の平均）と
「半径方向のずれ」（根元 → 片の向きの平均。負なら片より手前＝根元側）も表示する [mm]。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIST_EDGES_CM = (14, 15, 16, 18)  # 距離の区切り [cm]
COUNT_EDGES = (10, 15, 20)  # 片の数の区切り
REASONS = ("success", "not_grasped", "missed_pot", "wrong_piece")
REASON_LABELS = {
    "success": "成功",
    "not_grasped": "掴めず",
    "missed_pot": "鍋外",
    "wrong_piece": "別の片",
}


def _bin_labels(edges: tuple[float, ...], unit: str, integer: bool = False) -> list[str]:
    """区切り ``edges`` で分けた区間の名前. ``integer`` なら「5〜9個」のように両端を含めて書く."""
    if integer:
        labels = [f"〜{edges[0] - 1}{unit}"]
        labels += [f"{a}〜{b - 1}{unit}" for a, b in zip(edges, edges[1:], strict=False)]
        return labels + [f"{edges[-1]}{unit}〜"]
    labels = [f"〜{edges[0]}{unit}"]
    labels += [f"{a}〜{b}{unit}" for a, b in zip(edges, edges[1:], strict=False)]
    return labels + [f"{edges[-1]}{unit}〜"]


def _bin(value: float, edges: tuple[float, ...]) -> int:
    return sum(value >= e for e in edges)


def _table(
    rows: list[dict], key, edges: tuple[float, ...], title: str, unit: str, integer: bool = False
) -> list[str]:
    labels = _bin_labels(edges, unit, integer)
    groups: list[list[dict]] = [[] for _ in labels]
    for r in rows:
        groups[_bin(key(r), edges)].append(r)
    landing = "land_err_mm" in rows[0]
    header = f"  {title:<12} 回数  成功率  " + "  ".join(f"{REASON_LABELS[k]}" for k in REASONS[1:])
    if landing:
        header += "  着地のずれ  半径方向"
    lines = [header]
    for label, group in zip(labels, groups, strict=True):
        if not group:
            continue
        reasons = Counter(r["reason"] for r in group)
        rate = reasons["success"] / len(group)
        fails = "  ".join(f"{reasons[k]:>4}" for k in REASONS[1:])
        line = f"  {label:<12} {len(group):>4}  {rate:>5.0%}   {fails}"
        if landing:
            err = sum(float(r["land_err_mm"]) for r in group) / len(group)
            radial = sum(float(r["land_radial_mm"]) for r in group) / len(group)
            line += f"   {err:>7.1f}   {radial:>+7.1f}"
        lines.append(line)
    return lines


def _target_dist_cm(row: dict) -> float:
    return math.hypot(float(row["obj_x"]), float(row["obj_y"])) * 100


def summarize_eval(name: str, eval_dir: Path) -> list[str]:
    path = eval_dir / name / "episodes.csv"
    if not path.exists():
        return [f"== {name}: {path} がありません"]
    rows = list(csv.DictReader(path.open()))
    reasons = Counter(r["reason"] for r in rows)
    outcome = ", ".join(f"{REASON_LABELS.get(k, k)} {reasons[k]}" for k in REASONS if reasons[k])
    lines = [
        f"== {name}（{len(rows)} 回）成功率 {reasons['success'] / len(rows):.0%}  [{outcome}]",
        *_table(rows, _target_dist_cm, DIST_EDGES_CM, "掴む片の距離", "cm"),
    ]
    if "n_pieces" in rows[0]:
        lines += _table(rows, lambda r: int(r["n_pieces"]), COUNT_EDGES, "片の数", "個", True)
    return lines


def summarize_dataset(name: str, data_dir: Path) -> list[str]:
    path = data_dir / name / "episodes_meta.jsonl"
    if not path.exists():
        return [f"== 学習データ {name}: {path} がありません"]
    metas = [json.loads(line) for line in path.read_text().splitlines()]
    labels = _bin_labels(DIST_EDGES_CM, "cm")
    counts = Counter(_bin(math.hypot(*m["obj_pose0"][:2]) * 100, DIST_EDGES_CM) for m in metas)
    parts = [f"{labels[i]} {counts[i] / len(metas):.0%}" for i in range(len(labels))]
    return [
        f"== 学習データ {name}（{len(metas)} 本）掴んだ片までの距離の割合",
        "  " + "  ".join(parts),
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="+", help="評価の名前（outputs/eval/<名前>）")
    ap.add_argument("--dataset", default=None, help="学習データの名前（距離の分布を並べる）")
    ap.add_argument("--eval-dir", type=Path, default=REPO / "outputs" / "eval")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args(argv)
    lines = []
    for name in args.names:
        lines += summarize_eval(name, args.eval_dir) + [""]
    if args.dataset:
        lines += summarize_dataset(args.dataset, args.data_dir)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
