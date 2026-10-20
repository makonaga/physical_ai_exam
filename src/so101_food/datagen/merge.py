"""生成済みの学習データ（LeRobot v3 形式）を 1 つにまとめる.

    python -m so101_food.datagen.merge --name pot_lang_v2 pot_lang_v1 pot_lang_v1b

``data/<元の名前>/`` を並べた順に ``data/<name>/`` にまとめる（元のデータは消さない）。
``episodes_meta.jsonl`` は元のデータの順に並べてエピソード番号を振り直し、元のデータ名を
``source`` に書く。
``generation_stats.json`` には合計と、元のデータそれぞれの統計を書く。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def merge(names: list[str], name: str, data_dir: Path, overwrite: bool = False) -> Path:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    out = data_dir / name
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} は既にあります。作り直すなら --overwrite を付けてください。")
        shutil.rmtree(out)
    sources = [data_dir / n for n in names]
    for src in sources:
        if not (src / "meta" / "info.json").exists():
            raise SystemExit(f"{src} に学習データがありません。")
    print(f"{len(sources)} 個のデータをまとめます → {out}", flush=True)
    aggregate_datasets(
        repo_ids=[f"local/{n}" for n in names],
        aggr_repo_id=f"local/{name}",
        roots=sources,
        aggr_root=out,
    )

    metas, stats = [], {}
    for n, src in zip(names, sources, strict=True):
        lines = (src / "episodes_meta.jsonl").read_text().splitlines()
        metas += [{**json.loads(line), "source": n} for line in lines]
        stats[n] = json.loads((src / "generation_stats.json").read_text())
    with open(out / "episodes_meta.jsonl", "w") as f:
        for i, m in enumerate(metas):
            f.write(json.dumps({**m, "episode_index": i}) + "\n")
    attempts = sum(s["expert_attempts"] for s in stats.values())
    episodes = sum(s["episodes"] for s in stats.values())
    total = {
        "dataset": f"local/{name}",
        "task": stats[names[0]]["task"],
        "episodes": episodes,
        "frames": sum(s["frames"] for s in stats.values()),
        "expert_attempts": attempts,
        "expert_success_rate": round(episodes / max(attempts, 1), 4),
        "sources": {n: {k: v for k, v in s.items() if k != "seeds"} for n, s in stats.items()},
    }
    (out / "generation_stats.json").write_text(json.dumps(total, ensure_ascii=False, indent=2))

    merged = LeRobotDatasetMetadata(f"local/{name}", root=out)
    if merged.total_episodes != episodes or len(metas) != episodes:
        raise RuntimeError(
            f"エピソード数が合いません: まとめたデータ {merged.total_episodes}、"
            f"元のデータの合計 {episodes}、episodes_meta {len(metas)}"
        )
    summary = {k: v for k, v in total.items() if k != "sources"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"完了: {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sources", nargs="+", help="まとめる学習データの名前（data/<名前>）")
    ap.add_argument("--name", required=True, help="まとめたデータの名前")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--overwrite", action="store_true", help="同名のデータを消して作り直す")
    args = ap.parse_args(argv)
    merge(args.sources, args.name, args.data_dir, args.overwrite)


if __name__ == "__main__":
    main()
