"""学習済みポリシーの評価（CPU で並列）.

    python -m so101_food.eval.evaluate \\
        --policy outputs/train/act_pot_cube_v1/checkpoints/last/pretrained_model \\
        --name act_pot_cube_v1 --episodes 50 --workers 25
    # 段階2: 複数の食材片（学習で見せていない形）から一番近い片を鍋に入れる
    python -m so101_food.eval.evaluate --policy ... --name ... --task multi --piece-set carrot_test

ポリシーの入力は学習データと同じ（俯瞰・手首カメラ画像と 6 関節の角度）で、
物体の真の位置は使わない。各エピソードの結果を次のように分類する。

- ``success``     : 対象の小片（複数片のときはロボットに一番近い片）が鍋の中に入った
- ``wrong_piece`` : 対象ではない片を鍋に入れた・持ち上げた（複数片のとき）
- ``not_grasped`` : 対象の小片を持ち上げられなかった
- ``missed_pot``  : 持ち上げたが鍋に入らなかった（運搬中の落下・鍋の外に落とした）

複数片のときの配置は、学習データと同じく「エキスパートなら他の片に指を当てずに掴める」
ものだけを使う（エキスパートは配置の確認にだけ使い、ロボットは動かさない）。

``episodes.csv`` には、手先（TCP）が一番低くなった位置（＝掴もうとした位置）と、掴む片の初期位置との
ずれも記録する（``land_err_mm``: 水平距離、``land_radial_mm``: ロボットの根元から片へ向かう方向の
ずれで、正なら片より遠く・負なら手前、``land_tangent_mm``: それと直交する方向のずれ）。

結果は ``outputs/eval/<name>/`` に ``summary.json``・``episodes.csv``・動画として保存する。
学習データの seed（0 から数百）と重ならないよう、評価の seed は 100000 から始める。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import replace
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
    from so101_food.datagen.schema import from_dataset_units, to_dataset_units
    from so101_food.env.cameras import CameraRig
    from so101_food.env.factory import make_env
    from so101_food.env.pot_task import FRAME_SKIP, failure_reason
    from so101_food.env.spec import TaskSpec
    from so101_food.expert.pot_expert import PotExpert
    from so101_food.utils.render import save_mp4

    policy, pre, post = _load_policy(job["policy"], job["n_action_steps"], job["temporal_ensemble"])
    spec = TaskSpec(**job["task"])
    env = rig = None
    results = []
    for index, seed in job["episodes"]:
        # 食材片はエピソードごとに選んで環境を作る（どの片になるかはエピソード番号だけで
        # 決まるので、並列数を変えても同じ条件で評価できる）
        if env is None or spec.task != "cube":
            if rig is not None:
                rig.close()
                env.close()
            env = make_env(replace(spec, pool_size=1), np.random.default_rng([job["seed"], index]))
            env.scene_check = PotExpert(env).can_grasp
            rig = CameraRig(env.model)
            low, high = env.action_space.low, env.action_space.high
        t0 = time.time()
        env.reset(seed=seed)
        policy.reset()
        obj_pose0 = env.object_pose()
        record = index < job["videos"]
        frames, info, success_since = [], {}, None
        land_z, land_xy = np.inf, obj_pose0[:2].copy()  # 手先が一番低くなった高さと位置
        step = 0
        for step in range(1, job["max_steps"] + 1):
            images = rig.render(env.data.qpos)
            if record:
                frames.append(np.concatenate([images["overhead"], images["wrist"]], axis=1))
            obs = {
                "observation.state": torch.from_numpy(to_dataset_units(env.robot_qpos()))[None],
                "task": [env.task_description],
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
            tcp = env._get_tcp_pose()[:3]
            if tcp[2] < land_z:
                land_z, land_xy = float(tcp[2]), tcp[:2].copy()
            # 鍋に入ってから一定時間たったら打ち切る（そのまま入っていれば成功のまま）
            if info["success"]:
                success_since = step if success_since is None else success_since
                if step - success_since >= job["settle_steps"]:
                    break
            else:
                success_since = None
        reason = failure_reason(info)
        # 掴もうとした位置のずれを、根元 → 片の向き（半径方向）とそれに直交する向きに分ける
        radial_dir = obj_pose0[:2] / np.linalg.norm(obj_pose0[:2])
        offset = (land_xy - obj_pose0[:2]) * 1e3
        result = {
            "episode": index,
            "seed": seed,
            "object": info["target_object"],
            "n_pieces": int(info["n_pieces"]),
            "success": reason == "success",
            "reason": reason,
            "steps": step,
            "max_lift_m": round(float(info["max_lift"]), 3),
            "ever_grasped": bool(info["ever_grasped"]),
            "obj_x": round(float(obj_pose0[0]), 4),
            "obj_y": round(float(obj_pose0[1]), 4),
            "land_z_mm": round(land_z * 1e3, 1),
            "land_err_mm": round(float(np.linalg.norm(offset)), 1),
            "land_radial_mm": round(float(offset @ radial_dir), 1),
            "land_tangent_mm": round(float(offset @ [-radial_dir[1], radial_dir[0]]), 1),
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


def _success_by_count(results: list[dict], width: int = 5) -> dict:
    """机の上の片の数（5 個ずつの区間）ごとの成功率. キーは区間に入った片の数の最小〜最大."""
    groups: dict[int, list[dict]] = {}
    for r in results:
        groups.setdefault(r["n_pieces"] // width, []).append(r)
    out = {}
    for _, group in sorted(groups.items()):
        counts = [r["n_pieces"] for r in group]
        out[f"{min(counts)}-{max(counts)}"] = {
            "episodes": len(group),
            "success_rate": round(sum(r["success"] for r in group) / len(group), 4),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    from so101_food.env.spec import TASKS, TaskSpec
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
    ap.add_argument(
        "--task",
        choices=TASKS,
        default="cube",
        help="cube（段階1）/ pieces（食材片 1 個）/ multi（段階2: 複数の食材片）",
    )
    ap.add_argument(
        "--piece-set",
        default="carrot_test",
        help="食材片の形状セット（既定は学習で見せていない形）",
    )
    ap.add_argument("--cube-mm", type=float, default=25.0, help="立方体の 1 辺（分布外評価用）")
    ap.add_argument(
        "--spawn-radius", type=float, default=0.07, help="配置範囲の半径（分布外評価用）"
    )
    ap.add_argument("--min-pieces", type=int, default=5, help="multi で机に置く片の数の下限")
    ap.add_argument("--max-pieces", type=int, default=20, help="multi で机に置く片の数の上限")
    ap.add_argument("--n-action-steps", type=int, default=None, help="行動チャンクのうち実行する数")
    ap.add_argument(
        "--temporal-ensemble", type=float, default=None, help="ACT の時間的アンサンブル係数"
    )
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "eval")
    args = ap.parse_args(argv)
    task = TaskSpec(
        task=args.task,
        cube_mm=args.cube_mm,
        piece_set=args.piece_set,
        pool_size=1,  # 評価ではエピソードごとに選ぶ（multi では max_pieces 個）
        spawn_radius=args.spawn_radius,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
    )

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
            "task": task.to_dict(),
            "seed": args.seed,
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
        "by_n_pieces": _success_by_count(results) if task.task == "multi" else None,
        "task": task.to_dict(),
        "settings": {
            k: getattr(args, k)
            for k in ("seed", "max_steps", "n_action_steps", "temporal_ensemble")
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"完了: {out}", flush=True)


if __name__ == "__main__":
    main()
