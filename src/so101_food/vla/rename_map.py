"""事前学習済みポリシーのカメラ名に、学習データのカメラ名を対応付ける（--rename_map 用）.

    python -m so101_food.vla.rename_map --policy lerobot/smolvla_base --dataset pot_multi_v1

事前学習済みの VLA は、学習したときのカメラ名（例: ``observation.image``）で画像を受け取る。
このプロジェクトのデータのカメラ名（``observation.images.overhead`` /
``observation.images.wrist``）はそれと違うので、そのままでは lerobot-train が
「カメラ名が合わない」と止まる。そこで、

1. 事前学習済みポリシーの設定（config.json）から、画像入力の名前を順に読む
2. 名前に ``wrist`` を含むものがあれば手首カメラを、残りに俯瞰カメラを対応付ける
   （なければ設定に書かれた順に、俯瞰 → 手首の順で割り当てる）

結果を JSON（``{"データの名前": "ポリシーの名前", ...}``）で標準出力に、説明を標準エラーに出す。
名前がすでに合っていれば ``{}`` を出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# このプロジェクトのカメラ（データセットのキー）。俯瞰を先に割り当てる
DATASET_CAMERAS = ("observation.images.overhead", "observation.images.wrist")


def build_rename_map(expected: list[str], provided: list[str]) -> dict[str, str]:
    """ポリシーが受け取る画像名 ``expected`` に、データの画像名 ``provided`` を対応付ける."""
    if set(provided) <= set(expected):
        return {}
    if len(provided) > len(expected):
        raise ValueError(f"カメラが多すぎます: データ {provided}、ポリシー {expected}")
    mapping: dict[str, str] = {}
    free = list(expected)
    # 手首カメラは、名前で分かる場合はそれに合わせる
    for cam in provided:
        if "wrist" in cam:
            match = next((e for e in free if "wrist" in e), None)
            if match is not None:
                mapping[cam] = match
                free.remove(match)
    for cam in provided:
        if cam not in mapping:
            mapping[cam] = free.pop(0)
    return mapping


def main(argv: list[str] | None = None) -> None:
    import lerobot.policies  # noqa: F401  ポリシーの種類（smolvla など）を登録する
    from lerobot.configs import FeatureType
    from lerobot.configs.policies import PreTrainedConfig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_base", help="事前学習済みポリシー")
    ap.add_argument("--dataset", required=True, help="データセット名（data/<名前>）")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    args = ap.parse_args(argv)

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    expected = [k for k, ft in cfg.input_features.items() if ft.type is FeatureType.VISUAL]
    info = json.loads((args.data_dir / args.dataset / "meta" / "info.json").read_text())
    provided = [k for k in DATASET_CAMERAS if k in info["features"]]
    mapping = build_rename_map(expected, provided)
    print(f"{args.policy} の画像入力: {expected}", file=sys.stderr)
    for src, dst in mapping.items():
        print(f"  {src} → {dst}", file=sys.stderr)
    print(json.dumps(mapping))


if __name__ == "__main__":
    main()
