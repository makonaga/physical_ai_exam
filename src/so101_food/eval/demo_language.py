"""同じ配置で、指示文の食材だけを変えてポリシーを動かすデモ（段階3。指示文で動きが変わるかを見る）.

    python -m so101_food.eval.demo_language \\
        --policy outputs/train/smolvla_pot_lang_v1/checkpoints/last/pretrained_model \\
        --name smolvla_pot_lang_v1 --scenes 3

seed を順に試し、にんじん・じゃがいも・大根のどれを指示しても配置（机の上の片と位置）が同じで、
指示ごとに掴む片が違い、エキスパートならどれも掴める配置を探す。見つかった配置ごとに、
指示文を 3 通りに変えてポリシーを動かし、左上に指示文と結果を書いた動画を 1 本にまとめて
``outputs/demo/<name>/`` に保存する（形状セットは評価と同じく学習で見せていない形）。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from so101_food.eval.evaluate import VLA_TYPES, make_selector, run_policy_episode
from so101_food.eval.inference import NOISE_MODES

REPO = Path(__file__).resolve().parents[3]
FOODS = ("carrot", "potato", "daikon")
HOLD_FRAMES = 25  # 各指示の最後に結果を表示したまま止める長さ（1 秒）


def _caption(image, lines: list[str]):
    """画像の左上に、黒地に白い文字で複数行を書く."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=12)
    y = 3
    for text in lines:
        x0, y0, x1, y1 = draw.textbbox((4, y), text, font=font)
        draw.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=(0, 0, 0))
        draw.text((4, y), text, fill=(255, 255, 255), font=font)
        y = y1 + 4
    return np.asarray(pil)


def _find_scene(spec, foods: tuple[str, ...], seeds) -> tuple:
    """指示する食材を変えても配置が同じで、掴む片がすべて違う seed を探す. (環境, seed) を返す."""
    import numpy as np

    from so101_food.env.factory import make_env
    from so101_food.expert.pot_expert import PotExpert

    for seed in seeds:
        env = make_env(spec, np.random.default_rng([seed]))
        env.scene_check = PotExpert(env).can_grasp
        layouts, targets = set(), set()
        for food in foods:
            env.forced_food = food
            env.reset(seed=seed)
            layouts.add(env.layout_key())
            targets.add(env.active_slots[0])
        if len(layouts) == 1 and len(targets) == len(foods):
            return env, seed
        env.close()
    raise RuntimeError("条件に合う配置が見つかりませんでした")


def _demo_scene(job: dict) -> dict:
    import torch

    torch.set_num_threads(job["torch_threads"])
    from so101_food.env.cameras import CameraRig
    from so101_food.env.spec import TaskSpec
    from so101_food.utils.render import save_mp4

    select = make_selector(job)
    spec = replace(TaskSpec(**job["task"]), pool_size=1)
    start, stride = job["seed_start"], job["stride"]
    seeds = range(start, start + stride * job["max_tries"], stride)
    env, seed = _find_scene(spec, tuple(job["foods"]), seeds)
    rig = CameraRig(env.model)
    frames, results = [], []
    for food in job["foods"]:
        env.forced_food = food
        result, clip = run_policy_episode(
            env, rig, select, seed, job["max_steps"], job["settle_steps"], record=True
        )
        lines = [f"scene {job['scene']}  {env.task_description}"]
        frames += [_caption(f, lines) for f in clip]
        lines.append(f"result: {result['reason']}")
        frames += [_caption(clip[-1], lines)] * HOLD_FRAMES
        results.append({"food": food, **result})
        print(f"[demo] scene {job['scene']} seed={seed} {food}: {result['reason']}", flush=True)
    path = Path(job["out"]) / f"scene{job['scene']}_seed{seed}.mp4"
    save_mp4(frames, path, fps=25)
    rig.close()
    env.close()
    return {"scene": job["scene"], "seed": seed, "video": path.name, "results": results}


def main(argv: list[str] | None = None) -> None:
    from so101_food.env.spec import TaskSpec, piece_counts
    from so101_food.utils.parallel import prepare_workers, run_parallel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--policy", required=True, help="学習済みモデルのディレクトリ（pretrained_model）"
    )
    ap.add_argument("--name", required=True, help="結果の保存名（outputs/demo/<name>）")
    ap.add_argument("--scenes", type=int, default=3, help="配置の数（並列に動かす）")
    ap.add_argument("--foods", nargs="+", default=list(FOODS), choices=FOODS, help="指示する食材")
    ap.add_argument("--seed", type=int, default=300000, help="配置を探し始める seed")
    ap.add_argument("--max-tries", type=int, default=200, help="1 配置あたりに試す seed の数")
    ap.add_argument("--piece-set", default="test", help="形状セット（既定は学習で見せていない形）")
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--temporal-ensemble", type=float, default=None)
    ap.add_argument("--noise", choices=NOISE_MODES, default="random")
    ap.add_argument("--torch-threads", type=int, default=None, help="既定: VLA 3、それ以外 1")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "demo")
    args = ap.parse_args(argv)
    lo, hi = piece_counts("lang")
    task = TaskSpec(task="lang", piece_set=args.piece_set, min_pieces=lo, max_pieces=hi)

    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    policy = str(Path(args.policy).resolve())
    prepare_workers()
    import lerobot.policies  # noqa: F401  ポリシーの種類（act, smolvla など）を登録する
    from lerobot.configs.policies import PreTrainedConfig

    is_vla = PreTrainedConfig.from_pretrained(policy).type in VLA_TYPES
    if args.noise != "random" and not is_vla:
        raise SystemExit("--noise は VLA（flow matching で行動を作るモデル）だけで使えます")
    threads = args.torch_threads or (3 if is_vla else 1)
    jobs = [
        {
            "scene": k,
            "seed_start": args.seed + k,
            "stride": args.scenes,  # 配置ごとに別の seed を試す
            "max_tries": args.max_tries,
            "foods": args.foods,
            "policy": policy,
            "out": str(out),
            "task": task.to_dict(),
            "max_steps": args.max_steps,
            "settle_steps": args.settle_steps,
            "n_action_steps": args.n_action_steps,
            "temporal_ensemble": args.temporal_ensemble,
            "noise": args.noise,
            "torch_threads": threads,
        }
        for k in range(args.scenes)
    ]
    print(f"{args.scenes} 配置 × {len(args.foods)} 指示でデモを作ります: {policy}", flush=True)
    t0 = time.time()
    scenes = sorted(run_parallel(_demo_scene, jobs, args.scenes, "scene"), key=lambda s: s["scene"])
    summary = {
        "policy": policy,
        "scenes": scenes,
        "success": sum(r["success"] for s in scenes for r in s["results"]),
        "runs": sum(len(s["results"]) for s in scenes),
        "settings": {k: getattr(args, k) for k in ("noise", "temporal_ensemble", "n_action_steps")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for s in scenes:
        reasons = ", ".join(f"{r['food']} {r['reason']}" for r in s["results"])
        print(f"scene {s['scene']}（seed {s['seed']}）: {reasons} → {s['video']}", flush=True)
    print(f"完了: {out}", flush=True)


if __name__ == "__main__":
    main()
