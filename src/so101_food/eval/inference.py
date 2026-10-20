"""学習済みポリシーから 1 ステップ分の行動を出す（推論のしかたの違いをここにまとめる）.

- 既定: ポリシーの ``select_action``（行動チャンクを ``n_action_steps`` 個実行してから次を推論）
- ``noise``: flow matching で行動を作る VLA（SmolVLA など）の初期ノイズ。``"random"`` は毎回
  新しい乱数（LeRobot の既定）、``"zero"`` は 0、``"fixed"`` は毎回同じ乱数。推論し直すたびに違う
  ノイズから行動を作ると経路が切り替わって動きが揺れるので、固定するとどうなるかを比べる
- ``ensemble``: ``replan`` ステップごとに行動チャンクを推論し、同じ時刻への複数の予測を
  重み付き平均する（ACT の temporal ensemble と同じ重み ``exp(-ensemble * i)``。
  i = 0 が一番古い予測）。
  ACT は LeRobot の実装（毎ステップ推論）を使うので、ここでは VLA 用

torch はクラスを使うときに読み込む（``NOISE_MODES`` だけを使う側で torch を読み込まないように）。
"""

from __future__ import annotations

import math

NOISE_MODES = ("random", "zero", "fixed")


class ActionSelector:
    def __init__(
        self,
        policy,
        pre,
        post,
        noise: str = "random",
        ensemble: float | None = None,
        replan: int = 10,
    ):
        import torch

        if noise not in NOISE_MODES:
            raise ValueError(f"noise must be one of {NOISE_MODES}: {noise}")
        if ensemble is not None and not 1 <= replan <= policy.config.chunk_size:
            raise ValueError(f"replan must be 1..{policy.config.chunk_size}: {replan}")
        self.policy, self.pre, self.post = policy, pre, post
        self.ensemble, self.replan = ensemble, replan
        self.noise = None
        if noise != "random":
            cfg = policy.config
            shape = (1, cfg.chunk_size, cfg.max_action_dim)
            if noise == "zero":
                self.noise = torch.zeros(shape)
            else:
                self.noise = torch.randn(shape, generator=torch.Generator().manual_seed(0))
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.t = 0
        self.chunks: list = []  # (予測した時刻, 行動チャンク)。古い順

    def _kwargs(self) -> dict:
        return {} if self.noise is None else {"noise": self.noise}

    def __call__(self, obs: dict):
        """観測（1 ステップ分、バッチ 1）から行動（バッチ 1、ポリシーの出力の単位）を返す."""
        import torch

        with torch.inference_mode():
            return self._select(obs)

    def _select(self, obs: dict):
        batch = self.pre(obs)
        if self.ensemble is None:
            return self.post(self.policy.select_action(batch, **self._kwargs()))
        if self.t % self.replan == 0:
            chunk = self.policy.predict_action_chunk(batch, **self._kwargs())[0]
            self.chunks.append((self.t, chunk))
        # 今の時刻を含む予測だけを残し、古い順に重みを付けて平均する
        self.chunks = [(s, c) for s, c in self.chunks if self.t < s + len(c)]
        preds = [c[self.t - s] for s, c in self.chunks]
        weights = [math.exp(-self.ensemble * i) for i in range(len(preds))]
        action = sum(w * p for w, p in zip(weights, preds, strict=True)) / sum(weights)
        self.t += 1
        return self.post(action[None])
