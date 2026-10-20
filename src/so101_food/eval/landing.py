"""評価で手先を下ろした位置が、机の上のどの片に一番近かったかを調べる（指示文を使っているかの確認）.

    python -m so101_food.eval.landing smolvla_pot_lang_v1 act_pot_lang_v1 smolvla_pot_multi_v1

``outputs/eval/<名前>/`` の ``summary.json`` と ``episodes.csv`` から、評価と同じ seed で配置を
作り直し（ロボットは動かさない）、手先が一番低くなった位置（``land_*``）に一番近い片を次に分ける。

- 掴む片          : 指示どおりの片
- 同じ食材の別の片: 指示した食材だが、一番近い片ではない
- 別の食材の候補  : 指示と違う食材の「一番近い片」（その食材を指示されていたら掴むべき片）
- その他の片      : 上のどれでもない片

あわせて、手先が片の上（片の中心から 2cm 以内）に下りたか、片と片の間に下りたかと、
手先から「掴む片」「各食材の候補の真ん中」までの距離を出す。指示文を使わずに候補の間を狙っていれば、
候補の真ん中に近く、片の間に下りることが多くなる。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ON_PIECE_MM = 20.0  # 片の中心からこの距離以内なら「片の上に下りた」とみなす
CLASSES = ("target", "same_food", "other_candidate", "other")
LABELS = {
    "target": "掴む片",
    "same_food": "同じ食材の別の片",
    "other_candidate": "別の食材の候補",
    "other": "その他の片",
}


def _analyze_chunk(job: dict) -> list[dict]:
    import numpy as np

    from so101_food.env.factory import make_env
    from so101_food.env.spec import TaskSpec
    from so101_food.expert.pot_expert import PotExpert

    spec = replace(TaskSpec(**job["task"]), pool_size=1)
    out = []
    for row in job["rows"]:
        index, seed = int(row["episode"]), int(row["seed"])
        env = make_env(spec, np.random.default_rng([job["seed"], index]))
        env.scene_check = PotExpert(env).can_grasp
        env.reset(seed=seed)
        active = env.active_slots
        target = active[0]
        xy = {s: env.object_pose(s)[:2] for s in active}
        food = {s: env._slot_food(s) for s in active}
        dist0 = {s: float(np.linalg.norm(xy[s])) for s in active}
        candidates = {}  # 食材 → その食材で根元に一番近い片
        for s in sorted(active, key=dist0.get):
            candidates.setdefault(food[s], s)
        # 手先を下ろした位置を、評価で記録したずれ（根元 → 片の向きと、それに直交する向き）から戻す
        obj = np.array([float(row["obj_x"]), float(row["obj_y"])])
        if np.linalg.norm(xy[target] - obj) > 1e-3:  # 評価と同じ配置を作り直せているかの確認
            raise RuntimeError(f"ep{index}: 配置を作り直せませんでした（掴む片の位置が評価と違う）")
        radial = obj / np.linalg.norm(obj)
        tangent = np.array([-radial[1], radial[0]])
        offset = float(row["land_radial_mm"]) * radial + float(row["land_tangent_mm"]) * tangent
        land = obj + offset / 1e3
        nearest = min(active, key=lambda s: np.linalg.norm(xy[s] - land))
        others = [s for f, s in candidates.items() if f != food[target]]
        if nearest == target:
            cls = "target"
        elif food[nearest] == food[target]:
            cls = "same_food"
        elif nearest in others:
            cls = "other_candidate"
        else:
            cls = "other"
        centroid = np.mean([xy[s] for s in candidates.values()], axis=0)
        overall = min(active, key=dist0.get)  # 食材に関係なく根元に一番近い片（段階2の掴む片）
        out.append(
            {
                "episode": index,
                "reason": row["reason"],
                "class": cls,
                "on_piece": bool(np.linalg.norm(xy[nearest] - land) * 1e3 <= ON_PIECE_MM),
                "to_target_mm": float(np.linalg.norm(xy[target] - land) * 1e3),
                "to_centroid_mm": float(np.linalg.norm(centroid - land) * 1e3),
                "to_overall_nearest_mm": float(np.linalg.norm(xy[overall] - land) * 1e3),
                "on_overall_nearest": bool(nearest == overall),
                "target_is_overall_nearest": bool(target == overall),
                "target_to_centroid_mm": float(np.linalg.norm(centroid - xy[target]) * 1e3),
                "n_foods": len(candidates),
            }
        )
        env.close()
    return out


def _summary_lines(name: str, rows: list[dict]) -> list[str]:
    import numpy as np

    def pct(group, cond) -> str:
        return f"{sum(cond(r) for r in group) / len(group):4.0%}" if group else "  - "

    lines = [f"== {name}（{len(rows)} 回）手先を下ろした位置に一番近い片"]
    header = "  対象          回数  " + "  ".join(f"{LABELS[c]}" for c in CLASSES)
    lines.append(header + "  ｜ 片の上  片の間")
    failed = [r for r in rows if r["reason"] == "not_grasped"]
    for title, group in (("全部", rows), ("掴めず", failed)):
        cells = "  ".join(
            f"{pct(group, lambda r, c=c: r['class'] == c):>{len(LABELS[c]) * 2}s}" for c in CLASSES
        )
        on = pct(group, lambda r: r["on_piece"])
        off = pct(group, lambda r: not r["on_piece"])
        lines.append(f"  {title:<10s} {len(group):5d}  {cells}  ｜ {on:>6s}  {off:>6s}")
    multi = [r for r in rows if r["n_foods"] > 1]
    if multi:

        def med(key: str) -> float:
            return float(np.median([r[key] for r in multi]))

        lines.append(
            f"  中央値 [mm]: 手先 → 掴む片 {med('to_target_mm'):.1f}、"
            f"手先 → 候補の真ん中 {med('to_centroid_mm'):.1f}、"
            f"（参考）掴む片 → 候補の真ん中 {med('target_to_centroid_mm'):.1f}"
        )
        closer = sum(r["to_centroid_mm"] < r["to_target_mm"] for r in multi) / len(multi)
        lines.append(f"  手先が掴む片より候補の真ん中に近い回: {closer:.0%}")
        # 掴む片が「全体で一番近い片」と違う配置で、手先が全体で一番近い片に下りた割合（段階2の癖）
        differ = [r for r in multi if not r["target_is_overall_nearest"]]
        if differ:
            habit = sum(r["on_overall_nearest"] for r in differ) / len(differ)
            lines.append(
                f"  掴む片が全体で一番近い片ではない配置（{len(differ)} 回）で、"
                f"全体で一番近い片に下りた回: {habit:.0%}"
            )
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="+", help="評価の名前（outputs/eval/<名前>）")
    ap.add_argument("--eval-dir", type=Path, default=REPO / "outputs" / "eval")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    args = ap.parse_args(argv)
    # 配置を作り直すだけで描画はしない（子プロセスにも引き継ぐ）
    os.environ["MUJOCO_GL"] = "disabled"
    from so101_food.utils.parallel import run_parallel

    all_lines = []
    for name in args.names:
        d = args.eval_dir / name
        summary = json.loads((d / "summary.json").read_text())
        if summary["task"]["task"] not in ("multi", "lang"):
            raise SystemExit(f"{name}: 複数の片を置く評価（multi / lang）だけに使えます")
        rows = list(csv.DictReader((d / "episodes.csv").open()))
        if "land_radial_mm" not in rows[0]:
            raise SystemExit(f"{name}: 手先の着地位置が記録されていない評価です")
        workers = max(1, min(args.workers, len(rows)))
        jobs = [
            {"task": summary["task"], "seed": summary["settings"]["seed"], "rows": rows[w::workers]}
            for w in range(workers)
        ]
        results = sorted(
            (r for chunk in run_parallel(_analyze_chunk, jobs, workers, name) for r in chunk),
            key=lambda r: r["episode"],
        )
        (d / "landing.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        all_lines += _summary_lines(name, results) + [""]
    print("\n".join(all_lines))


if __name__ == "__main__":
    main()
