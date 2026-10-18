"""学習済みポリシーの評価（CPU で並列）.

    python -m so101_food.eval.evaluate \\
        --policy outputs/train/act_pot_cube_v1/checkpoints/last/pretrained_model \\
        --name act_pot_cube_v1 --episodes 50 --workers 25

ポリシーの入力は学習データと同じ（俯瞰・手首カメラ画像と 6 関節の角度）で、
物体の真の位置は使わない。各エピソードの結果を次のように分類する。

- ``success``     : 小片が鍋の中に入った
- ``not_grasped`` : 小片を持ち上げられなかった
- ``missed_pot``  : 持ち上げたが鍋に入らなかった（運搬中の落下・鍋の外に落とした）

結果は ``outputs/eval/<name>/`` に ``summary.json``・``episodes.csv``・動画として保存する。
学習データの seed（0 から数百）と重ならないよう、評価の seed は 100000 から始める。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_policy(path: str, n_action_steps: int | None, temporal_ensemble: float | None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = "cpu"
    if temporal_ensemble is not None:
        cfg.temporal_ensemble_coeff = temporal_ensemble
        cfg.n_action_steps = 1
    elif n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    return policy, pre, post


def _evaluate_chunk(job: dict) -> list[dict]:
    import numpy as np
    import torch

    torch.set_num_threads(job["torch_threads"])
    from so101_nexus import CubeObject

    from so101_food.datagen.schema import from_dataset_units, to_dataset_units
    from so101_food.env.cameras import CameraRig
    from so101_food.env.pot_task import (
        FRAME_SKIP,
        TASK_DESCRIPTION,
        PotTaskEnv,
        default_pick_config,
        failure_reason,
    )
    from so101_food.utils.render import save_mp4

    policy, pre, post = _load_policy(job["policy"], job["n_action_steps"], job["temporal_ensemble"])
    config = default_pick_config(
        objects=CubeObject(side_length_mm=job["cube_mm"], color="orange"),
        spawn_radius=job["spawn_radius"],
    )
    env = PotTaskEnv(config=config)
    rig = CameraRig(env.model)
    low, high = env.action_space.low, env.action_space.high
    results = []
    for index, seed in job["episodes"]:
        t0 = time.time()
        env.reset(seed=seed)
        policy.reset()
        obj_pose0 = env.object_pose()
        record = index < job["videos"]
        frames, info, success_since = [], {}, None
        step = 0
        for step in range(1, job["max_steps"] + 1):
            images = rig.render(env.data.qpos)
            if record:
                frames.append(np.concatenate([images["overhead"], images["wrist"]], axis=1))
            obs = {
                "observation.state": torch.from_numpy(to_dataset_units(env.robot_qpos()))[None],
                "task": [TASK_DESCRIPTION],
            }
            for key, img in images.items():
                obs[f"observation.images.{key}"] = (
                    torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None]
                )
            with torch.inference_mode():
                action = post(policy.select_action(pre(obs)))
            target = np.clip(from_dataset_units(action.squeeze(0).cpu().numpy()), low, high)
            for _ in range(FRAME_SKIP):
                _, _, _, _, info = env.step(target)
            # 鍋に入ってから一定時間たったら打ち切る（そのまま入っていれば成功のまま）
            if info["success"]:
                success_since = step if success_since is None else success_since
                if step - success_since >= job["settle_steps"]:
                    break
            else:
                success_since = None
        reason = failure_reason(info)
        result = {
            "episode": index,
            "seed": seed,
            "success": reason == "success",
            "reason": reason,
            "steps": step,
            "max_lift_m": round(float(info["max_lift"]), 3),
            "ever_grasped": bool(info["ever_grasped"]),
            "obj_x": round(float(obj_pose0[0]), 4),
            "obj_y": round(float(obj_pose0[1]), 4),
            "time_s": round(time.time() - t0, 1),
        }
        if record:
            name = f"ep{index:03d}_seed{seed}_{reason}.mp4"
            save_mp4(frames, Path(job["out"]) / "videos" / name, fps=25)
        print(f"[eval] ep{index:03d} seed={seed} {reason} ({result['time_s']}s)", flush=True)
        results.append(result)
    rig.close()
    env.close()
    return results


def main(argv: list[str] | None = None) -> None:
    from so101_food.utils.parallel import default_workers, prepare_workers, run_parallel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--policy", required=True, help="学習済みモデルのディレクトリ（pretrained_model）"
    )
    ap.add_argument("--name", required=True, help="結果の保存名（outputs/eval/<name>）")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=100000, help="評価の seed の開始値")
    ap.add_argument("--videos", type=int, default=6, help="動画を保存するエピソード数（先頭から）")
    ap.add_argument(
        "--max-steps", type=int, default=260, help="1 エピソードの上限（25Hz、既定 10.4 秒）"
    )
    ap.add_argument("--settle-steps", type=int, default=25, help="鍋に入ってから打ち切るまで")
    ap.add_argument("--cube-mm", type=float, default=25.0, help="立方体の 1 辺（分布外評価用）")
    ap.add_argument(
        "--spawn-radius", type=float, default=0.07, help="配置範囲の半径（分布外評価用）"
    )
    ap.add_argument("--n-action-steps", type=int, default=None, help="行動チャンクのうち実行する数")
    ap.add_argument(
        "--temporal-ensemble", type=float, default=None, help="ACT の時間的アンサンブル係数"
    )
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "eval")
    args = ap.parse_args(argv)

    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    policy = str(Path(args.policy).resolve())
    prepare_workers()
    workers = max(1, min(args.workers, args.episodes))
    episodes = [(i, args.seed + i) for i in range(args.episodes)]
    jobs = [
        {
            "episodes": episodes[w::workers],
            "policy": policy,
            "out": str(out),
            "videos": args.videos,
            "max_steps": args.max_steps,
            "settle_steps": args.settle_steps,
            "cube_mm": args.cube_mm,
            "spawn_radius": args.spawn_radius,
            "n_action_steps": args.n_action_steps,
            "temporal_ensemble": args.temporal_ensemble,
            "torch_threads": args.torch_threads,
        }
        for w in range(workers)
    ]
    print(f"{args.episodes} エピソードを {workers} プロセスで評価します: {policy}", flush=True)
    t0 = time.time()
    results = sorted(
        (r for chunk in run_parallel(_evaluate_chunk, jobs, workers, "worker") for r in chunk),
        key=lambda r: r["episode"],
    )
    with open(out / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    outcomes = Counter(r["reason"] for r in results)
    successes = [r for r in results if r["success"]]
    summary = {
        "policy": policy,
        "episodes": len(results),
        "success_rate": round(len(successes) / len(results), 4),
        "outcomes": dict(outcomes),
        "mean_steps_success": (
            round(sum(r["steps"] for r in successes) / len(successes), 1) if successes else None
        ),
        "settings": {
            k: getattr(args, k)
            for k in ("seed", "max_steps", "cube_mm", "spawn_radius", "n_action_steps")
        }
        | {"temporal_ensemble": args.temporal_ensemble},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"完了: {out}", flush=True)


if __name__ == "__main__":
    main()
