"""指示文だけを変えたとき、ポリシーの出す動きがどれだけ変わるかを調べる（言語を使っているかの確認）.

    python -m so101_food.eval.probe_language \\
        --policy outputs/train/smolvla_pot_lang_v1/checkpoints/last/pretrained_model \\
        --dataset pot_lang_v1 --episodes 50 --device cuda

学習データのエピソードから場面（``--frames`` のフレーム。0 はロボットが開始姿勢で、片を置いた直後）
を取り出し、次の 4 通りで行動チャンク（50 ステップ = 2 秒分）を出して、チャンクの最後の手先（TCP）の
位置を比べる。

- 本当の指示文（初期ノイズ 0）… エキスパートの行動との差（学習データをどれだけ再現できているか）
- 食材だけを他の 2 つに変えた指示文（初期ノイズ 0）… 本当の指示文との差（指示文の影響）
- 本当の指示文で初期ノイズだけを乱数に変える（VLA のみ）… 本当の指示文との差（ばらつき。比べる基準）

指示文を変えたときの差が、ノイズを変えたときの差と同じくらい小さければ、指示文をほとんど使っていない。
結果は画面と ``outputs/probe/<name>.json`` に出す。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FOODS = ("carrot", "potato", "daikon")


def _load_policy(path: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if getattr(cfg, "load_vlm_weights", False):
        cfg.load_vlm_weights = False  # VLM の重みも学習済みモデルに含まれている
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post


def _other_task(task: str, food: str) -> str:
    """指示文の食材名だけを ``food`` に置き換える."""
    return re.sub(r"\b(" + "|".join(FOODS) + r")\b", food, task, count=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--policy", required=True, help="学習済みモデルのディレクトリ（pretrained_model）"
    )
    ap.add_argument("--dataset", required=True, help="学習データの名前（data/<名前>）")
    ap.add_argument("--name", default=None, help="結果の保存名（既定: 学習の実行名）")
    ap.add_argument("--episodes", type=int, default=50, help="調べるエピソード数（等間隔に選ぶ）")
    ap.add_argument(
        "--frames", type=int, nargs="+", default=[0, 30], help="エピソードの何フレーム目か"
    )
    ap.add_argument("--device", default="cuda", help="cuda または cpu")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "probe")
    args = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "disabled")  # 手先の位置の計算だけで、描画はしない

    import mujoco
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_food.datagen.schema import from_dataset_units
    from so101_food.env.factory import make_env
    from so101_food.env.spec import TaskSpec
    from so101_food.eval.evaluate import VLA_TYPES

    policy_path = str(Path(args.policy).resolve())
    name = args.name or Path(policy_path).parts[-4]  # outputs/train/<実行名>/checkpoints/...
    policy, pre, post = _load_policy(policy_path, args.device)
    is_vla = policy.config.type in VLA_TYPES
    ds = LeRobotDataset(f"local/{args.dataset}", root=args.data_dir / args.dataset)

    env = make_env(
        TaskSpec(task="cube"), np.random.default_rng(0)
    )  # 手先の位置の計算用（ロボットは同じ）
    fk = mujoco.MjData(env.model)

    def tcp(row) -> np.ndarray:
        fk.qpos[env.robot_qpos_addr] = from_dataset_units(np.asarray(row))
        mujoco.mj_kinematics(env.model, fk)
        return fk.site_xpos[env._tcp_site_id].copy()

    def chunk(batch: dict, task: str, noise) -> np.ndarray:
        """行動チャンク（ステップ数, 6）をデータセットの単位で返す."""
        b = dict(batch, task=[task])
        kwargs = {} if noise is None else {"noise": noise}
        with torch.inference_mode():
            pred = policy.predict_action_chunk(pre(b), **kwargs)
            return np.stack(
                [post(pred[:, t]).float().cpu().numpy()[0] for t in range(pred.shape[1])]
            )

    shape = (1, policy.config.chunk_size, getattr(policy.config, "max_action_dim", 0))
    zero = torch.zeros(shape, device=args.device) if is_vla else None
    rand = (
        torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(args.device)
        if is_vla
        else None
    )

    episodes = np.linspace(0, ds.meta.total_episodes - 1, args.episodes).round().astype(int)
    rows = []
    for e in sorted(set(episodes.tolist())):
        ep = ds.meta.episodes[e]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        for f in args.frames:
            i = start + f
            if i >= end - 1:
                continue
            item = ds[i]
            batch = {
                k: v[None].to(args.device)
                for k, v in item.items()
                if k.startswith("observation.") and isinstance(v, torch.Tensor)
            }
            task = item["task"]
            food = next((x for x in FOODS if x in task), None)
            pred = chunk(batch, task, zero)
            n = min(len(pred), end - i)  # エピソードの終わりを越えない
            expert = torch.stack([torch.as_tensor(a) for a in ds.hf_dataset[i : i + n]["action"]])
            end_pred, end_expert = tcp(pred[n - 1]), tcp(expert[n - 1].numpy())
            row = {
                "episode": e,
                "frame": f,
                "food": food,
                # 手先がチャンクの間に動く距離（エキスパート）。下の差の大きさを見る目安
                "expert_move_mm": float(np.linalg.norm(end_expert - tcp(expert[0].numpy())) * 1e3),
                "vs_expert_mm": float(np.linalg.norm(end_pred - end_expert) * 1e3),
                "vs_expert_deg": float(np.abs(pred[:n, :5] - expert[:n, :5].numpy()).mean()),
            }
            if food is not None:
                changes = [
                    np.linalg.norm(tcp(chunk(batch, _other_task(task, o), zero)[n - 1]) - end_pred)
                    for o in FOODS
                    if o != food
                ]
                row["other_task_mm"] = float(np.mean(changes) * 1e3)
            if is_vla:
                row["other_noise_mm"] = float(
                    np.linalg.norm(tcp(chunk(batch, task, rand)[n - 1]) - end_pred) * 1e3
                )
            rows.append(row)
            print(
                f"[probe] ep{e:04d} frame {f:3d} {food}: "
                + "  ".join(f"{k} {v:.1f}" for k, v in row.items() if k.endswith(("_mm", "_deg"))),
                flush=True,
            )

    keys = [k for k in rows[0] if k.endswith(("_mm", "_deg"))]
    summary = {"policy": policy_path, "dataset": args.dataset, "by_frame": {}}
    print(f"\n== {name}（{args.dataset} の {len(set(r['episode'] for r in rows))} エピソード）")
    print("  フレーム  " + "  ".join(f"{k:>16s}" for k in keys) + "   （中央値）")
    for f in args.frames:
        group = [r for r in rows if r["frame"] == f]
        if not group:
            continue
        med = {k: float(np.median([r[k] for r in group if k in r])) for k in keys}
        summary["by_frame"][str(f)] = med
        print(f"  {f:8d}  " + "  ".join(f"{med[k]:16.1f}" for k in keys))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{name}.json"
    path.write_text(json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(
        "\n  expert_move_mm: エキスパートの手先がチャンクの間に動く距離\n"
        "  vs_expert_mm  : 本当の指示文で出した動きの最後の手先と、エキスパートの手先の距離\n"
        "  vs_expert_deg : 同じく関節角度の平均の差 [deg]\n"
        "  other_task_mm : 食材だけを変えた指示文にしたとき、最後の手先がどれだけ動くか\n"
        "  other_noise_mm: 初期ノイズだけを変えたとき、最後の手先がどれだけ動くか"
        "（VLA のみ。比べる基準）"
    )
    print(f"完了: {path}", flush=True)


if __name__ == "__main__":
    main()
