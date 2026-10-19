"""生成したデータセットの中身を、見やすい動画に書き出す.

    python -m so101_food.datagen.export_videos --dataset pot_multi_v1

LeRobot のデータセットでは、全エピソードの映像がカメラごとに 1 本につながった AV1 の動画に
なっていて、そのままだと見にくい（ブラウザによっては再生もできない）。そこで次を書き出す。

- ``ep<番号>_<物体名 or 片の数>.mp4``: エピソード 1 本ずつ。左が俯瞰カメラ、右が手首カメラ
- ``grid_<N>.mp4``: N 本のエピソードの俯瞰カメラを格子状に並べて同時に流す（ばらつきの確認用）

どちらも H.264 で、画面左上にエピソード番号・物体名（複数片のときは片の数）・経過時間を入れる。
出力先は ``outputs/dataset_videos/<データセット名>/``。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CAMERA_KEYS = ("observation.images.overhead", "observation.images.wrist")


def _to_uint8(chw) -> np.ndarray:
    return (chw.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=12)
    x0, y0, x1, y1 = draw.textbbox((4, 3), text, font=font)
    draw.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def _episode_frames(ds, episode: int) -> dict[str, list[np.ndarray]]:
    info = ds.meta.episodes[episode]
    frames: dict[str, list[np.ndarray]] = {key: [] for key in CAMERA_KEYS}
    for i in range(info["dataset_from_index"], info["dataset_to_index"]):
        item = ds[i]
        for key in CAMERA_KEYS:
            frames[key].append(_to_uint8(item[key]))
    return frames


def _short_name(meta: dict) -> str:
    """episodes_meta.jsonl から表示用の短い名前を作る.

    複数片のときは片の数（例: "12pieces"）、1 個のときは物体名（"carrot piece carrot_0042" →
    "carrot_0042"）。
    """
    if meta.get("n_pieces", 1) > 1:
        # 段階3（言葉で食材を指定）では、指定した食材名も付ける（例: "12pieces_potato"）
        food = meta.get("target_food")
        language = food is not None and food in meta.get("task", "")
        return f"{meta['n_pieces']}pieces" + (f"_{food}" if language else "")
    return meta.get("object", "cube").split()[-1]


def main(argv: list[str] | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_food.utils.render import save_mp4

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True, help="データセット名（data/<名前>）")
    ap.add_argument(
        "--episodes", type=int, nargs="*", help="書き出すエピソード番号（省略時は等間隔）"
    )
    ap.add_argument("--num", type=int, default=6, help="エピソードを省略したときに書き出す本数")
    ap.add_argument("--grid", type=int, default=9, help="一覧動画に並べる本数（0 なら作らない）")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "dataset_videos")
    args = ap.parse_args(argv)

    root = args.data_dir / args.dataset
    ds = LeRobotDataset(f"local/{args.dataset}", root=root)
    fps = ds.fps
    meta_path = root / "episodes_meta.jsonl"
    metas = (
        [json.loads(line) for line in meta_path.read_text().splitlines()]
        if meta_path.exists()
        else []
    )
    meta_of = {m["episode_index"]: m for m in metas}
    out = args.out_dir / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    n_total = ds.num_episodes
    print(f"{args.dataset}: {n_total} エピソード, {ds.num_frames} フレーム, {fps}fps", flush=True)

    episodes = args.episodes or np.linspace(0, n_total - 1, min(args.num, n_total)).round().astype(
        int
    )
    for ep in episodes:
        ep = int(ep)
        name = _short_name(meta_of.get(ep, {}))
        frames = _episode_frames(ds, ep)
        video = []
        for t, (top, wrist) in enumerate(zip(*frames.values(), strict=True)):
            text = f"ep {ep}  {name}  {t / fps:4.1f}s"
            video.append(_label(np.concatenate([top, wrist], axis=1), text))
        path = save_mp4(video, out / f"ep{ep:03d}_{name}.mp4", fps=fps)
        print(f"  {path.name}（{len(video)} フレーム）", flush=True)

    if args.grid > 0:
        grid_eps = np.linspace(0, n_total - 1, min(args.grid, n_total)).round().astype(int)
        tiles = []
        for ep in grid_eps:
            top = _episode_frames(ds, int(ep))[CAMERA_KEYS[0]]
            name = _short_name(meta_of.get(int(ep), {}))
            tiles.append([_label(f, f"ep {int(ep)} {name}") for f in top])
        cols = math.ceil(math.sqrt(len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        length = max(len(t) for t in tiles)
        blank = np.zeros_like(tiles[0][0])
        video = []
        for t in range(length):
            # 短いエピソードは最後のフレームのまま止めておく
            cells = [tile[min(t, len(tile) - 1)] for tile in tiles]
            cells += [blank] * (rows * cols - len(cells))
            row_images = [
                np.concatenate(cells[r * cols : (r + 1) * cols], axis=1) for r in range(rows)
            ]
            video.append(np.concatenate(row_images, axis=0))
        path = save_mp4(video, out / f"grid_{len(tiles)}.mp4", fps=fps)
        print(f"  {path.name}（{len(tiles)} 本を並べた一覧）", flush=True)
    print(f"完了: {out}", flush=True)


if __name__ == "__main__":
    main()
