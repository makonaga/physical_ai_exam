"""成功フィルタ付きの学習データ生成（並列）.

    python -m so101_food.datagen.generate --name pot_cube_v1 --episodes 300 --workers 36
    python -m so101_food.datagen.generate --name pot_multi_v1 --task multi --episodes 1000

各ワーカーは次を繰り返し、自分の割り当て数だけ成功エピソードを集める。

1. エキスパートを描画なしで実行し、状態列（qpos）と行動列を記録する（速い）
2. 成功したエピソードだけ、記録した状態列からカメラ画像を描画し直す（遅い）
3. LeRobot v3 形式のシャードに保存する

最後に全ワーカーのシャードを 1 つのデータセットに統合し、生成の統計
（試行数・成功率・失敗の内訳）を ``generation_stats.json`` に書く。
シャードは一時ディレクトリ（既定: ホーム配下のキャッシュ）に作り、統合後に消す。

``--task pieces`` / ``--task multi``（段階2）では、形状セットから ``--pool-size`` 個の食材片を
選んで環境を作り、``--rebuild-every`` 回の試行ごとに選び直す（多くの形を少しずつ使うため）。
``multi`` では各エピソードで ``--min-pieces``〜``--max-pieces`` 個を机に置き、ロボットに
一番近い片を鍋に入れる。配置はエキスパートが他の片に指を当てずに掴めるものだけを使う。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _quiet_libraries() -> None:
    """動画書き込み（ffmpeg）と datasets の進捗表示を止め、ログを読みやすくする."""
    import av.logging
    import datasets

    av.logging.set_level(av.logging.ERROR)
    datasets.disable_progress_bars()


def _generate_shard(job: dict) -> dict:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_food.datagen.schema import ROBOT_TYPE, dataset_features, to_dataset_units
    from so101_food.env.cameras import CameraRig
    from so101_food.env.factory import make_env
    from so101_food.env.pot_task import CONTROL_HZ
    from so101_food.env.spec import TaskSpec
    from so101_food.expert.pot_expert import PotExpert, run_expert_episode

    _quiet_libraries()
    t0 = time.time()
    spec = TaskSpec(**job["task"])
    env = rig = None

    def rebuild(count: int):
        nonlocal env, rig
        if rig is not None:
            rig.close()
            env.close()
        # 食材片の選び方もワーカーと作り直しの回数から決まるので、生成は再現できる
        env = make_env(spec, np.random.default_rng([job["seed_start"], count]))
        rig = CameraRig(env.model)
        new_expert = PotExpert(env)
        # 複数片のとき、エキスパートが掴めない配置（指が他の片に当たるなど）は reset で置き直す
        env.scene_check = new_expert.can_grasp
        return new_expert

    expert = rebuild(0)
    ds = LeRobotDataset.create(
        repo_id=job["repo_id"],
        fps=CONTROL_HZ,
        features=dataset_features(),
        root=job["root"],
        robot_type=ROBOT_TYPE,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=1,
    )
    reasons: Counter = Counter()
    episodes = []
    k = 0
    while len(episodes) < job["quota"] and k < job["max_attempts"]:
        if spec.task != "cube" and k > 0 and k % job["rebuild_every"] == 0:
            expert = rebuild(k // job["rebuild_every"])
        seed = job["seed_start"] + k * job["seed_stride"]
        k += 1
        rec = run_expert_episode(env, expert, seed)
        reasons[rec.reason] += 1
        if not rec.success:
            continue
        states, actions = to_dataset_units(rec.state), to_dataset_units(rec.action)
        for t in range(rec.n_frames):
            images = rig.render(rec.qpos[t])
            frame = {
                "observation.state": states[t],
                "action": actions[t],
                "task": env.task_description,
            }
            frame.update({f"observation.images.{key}": img for key, img in images.items()})
            ds.add_frame(frame)
        ds.save_episode()
        episodes.append(
            {
                "seed": seed,
                "object": rec.info["target_object"],
                "n_pieces": int(rec.info["n_pieces"]),
                "n_frames": rec.n_frames,
                "obj_pose0": np.round(rec.obj_pose0, 4).tolist(),
            }
        )
        print(
            f"[worker {job['worker']:02d}] {len(episodes)}/{job['quota']} "
            f"(試行 {k}, {time.time() - t0:.0f}s)",
            flush=True,
        )
    ds.finalize()
    rig.close()
    env.close()
    return {
        "worker": job["worker"],
        "root": job["root"],
        "repo_id": job["repo_id"],
        "attempts": k,
        "reasons": dict(reasons),
        "episodes": episodes,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> None:
    from so101_food.env.spec import TASKS, TaskSpec
    from so101_food.utils.parallel import default_workers, prepare_workers, run_parallel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", default="pot_cube_v1", help="データセット名")
    ap.add_argument("--episodes", type=int, default=300, help="集める成功エピソード数")
    ap.add_argument("--workers", type=int, default=default_workers(), help="並列プロセス数")
    ap.add_argument("--seed", type=int, default=0, help="seed の開始値")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data", help="保存先の親ディレクトリ")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home() / ".cache" / "so101_food" / "shards",
        help="シャードの一時保存先（ローカルディスク推奨）",
    )
    ap.add_argument("--overwrite", action="store_true", help="同名のデータセットを消して作り直す")
    ap.add_argument(
        "--task",
        choices=TASKS,
        default="cube",
        help="cube（段階1）/ pieces（食材片 1 個）/ multi（段階2: 複数の食材片）",
    )
    ap.add_argument(
        "--piece-set", default="carrot_train", help="食材片の形状セット（pieces / multi のとき）"
    )
    ap.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="1 つの環境に入れる食材片の数（既定: pieces 20、multi 24）",
    )
    ap.add_argument("--rebuild-every", type=int, default=10, help="食材片を選び直す試行回数の間隔")
    ap.add_argument("--min-pieces", type=int, default=5, help="multi で机に置く片の数の下限")
    ap.add_argument("--max-pieces", type=int, default=20, help="multi で机に置く片の数の上限")
    args = ap.parse_args(argv)
    pool_size = args.pool_size or (24 if args.task == "multi" else 20)
    task = TaskSpec(
        task=args.task,
        piece_set=args.piece_set,
        pool_size=pool_size,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
    )

    out = args.out_dir / args.name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} は既にあります。作り直すなら --overwrite を付けてください。")
        shutil.rmtree(out)
    work = args.work_dir / args.name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    gl = prepare_workers()
    workers = max(1, min(args.workers, args.episodes))
    base, extra = divmod(args.episodes, workers)
    quotas = [base + (1 if w < extra else 0) for w in range(workers)]
    jobs = [
        {
            "worker": w,
            "quota": quotas[w],
            "max_attempts": quotas[w] * 3 + 5,
            # ワーカー w の k 回目の試行は seed = seed + w + k * workers（全体で重複しない）
            "seed_start": args.seed + w,
            "seed_stride": workers,
            "task": task.to_dict(),
            "rebuild_every": args.rebuild_every,
            "repo_id": f"local/{args.name}_shard{w:03d}",
            "root": str(work / f"shard{w:03d}"),
        }
        for w in range(workers)
    ]
    print(f"成功 {args.episodes} エピソードを {workers} プロセスで生成します", flush=True)
    print(f"描画: {gl.renderer}", flush=True)
    t0 = time.time()
    results = run_parallel(_generate_shard, jobs, workers, "shard")
    results.sort(key=lambda r: r["worker"])
    t_gen = time.time() - t0

    from lerobot.datasets.aggregate import aggregate_datasets

    shards = [r for r in results if r["episodes"]]
    print(f"{len(shards)} 個のシャードを統合します → {out}", flush=True)
    aggregate_datasets(
        repo_ids=[r["repo_id"] for r in shards],
        aggr_repo_id=f"local/{args.name}",
        roots=[Path(r["root"]) for r in shards],
        aggr_root=out,
    )
    shutil.rmtree(work, ignore_errors=True)

    reasons: Counter = Counter()
    for r in results:
        reasons.update(r["reasons"])
    attempts = sum(r["attempts"] for r in results)
    n_success = sum(len(r["episodes"]) for r in results)
    stats = {
        "dataset": f"local/{args.name}",
        "task": task.to_dict(),
        "episodes": n_success,
        "frames": sum(e["n_frames"] for r in results for e in r["episodes"]),
        "expert_attempts": attempts,
        "expert_success_rate": round(n_success / max(attempts, 1), 4),
        "expert_outcomes": dict(reasons),
        "workers": workers,
        "renderer": gl.renderer,
        "generation_time_s": round(t_gen, 1),
        "total_time_s": round(time.time() - t0, 1),
        "seeds": sorted(e["seed"] for r in results for e in r["episodes"]),
    }
    (out / "generation_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    # 統合後のエピソード番号は、シャード（ワーカー番号）順・保存順に並ぶ
    with open(out / "episodes_meta.jsonl", "w") as f:
        episodes = [e for r in shards for e in r["episodes"]]
        for i, e in enumerate(episodes):
            f.write(json.dumps({"episode_index": i, **e}) + "\n")
    summary = {k: v for k, v in stats.items() if k != "seeds"}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"完了: {out}", flush=True)


if __name__ == "__main__":
    main()
