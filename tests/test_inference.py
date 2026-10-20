"""推論のしかた（ノイズの固定・行動チャンクの平均）の動作確認. 本物のモデルの代わりに偽物を使う."""

from types import SimpleNamespace

import pytest
import torch

from so101_food.eval.inference import ActionSelector

CHUNK, DIM = 6, 2


class FakePolicy:
    """呼ばれた回数 k に応じて、時刻 t の行動が [k, t] になる行動チャンクを返す."""

    def __init__(self):
        self.config = SimpleNamespace(chunk_size=CHUNK, max_action_dim=4)
        self.calls, self.noises = 0, []

    def reset(self):
        self.queue = []

    def predict_action_chunk(self, batch, noise=None):
        self.noises.append(noise)
        self.calls += 1
        t = torch.arange(CHUNK, dtype=torch.float32)
        return torch.stack([torch.full((CHUNK,), float(self.calls)), t], dim=1)[None]

    def select_action(self, batch, noise=None):
        if not self.queue:
            self.queue = list(self.predict_action_chunk(batch, noise)[0])
        return self.queue.pop(0)[None]


def identity(x):
    return x


def test_default_uses_select_action_and_random_noise():
    policy = FakePolicy()
    select = ActionSelector(policy, identity, identity)
    actions = [select({}) for _ in range(CHUNK + 1)]
    assert policy.calls == 2  # チャンクを出し切ってから次を推論する
    assert actions[0].shape == (1, DIM)
    assert policy.noises == [None, None]  # ノイズは LeRobot に任せる


@pytest.mark.parametrize("mode", ["zero", "fixed"])
def test_fixed_noise_is_the_same_every_time(mode):
    policy = FakePolicy()
    select = ActionSelector(policy, identity, identity, noise=mode)
    for _ in range(2 * CHUNK):
        select({})
    first, second = policy.noises
    assert first.shape == (1, CHUNK, 4)
    assert torch.equal(first, second)
    assert (first == 0).all() == (mode == "zero")


def test_ensemble_averages_overlapping_chunks():
    policy = FakePolicy()
    select = ActionSelector(policy, identity, identity, ensemble=0.0, replan=2)
    out = [select({})[0] for _ in range(5)]
    assert policy.calls == 3  # t = 0, 2, 4 で推論
    # t=0: 1 回目だけ / t=2: 1・2 回目の平均 / t=4: 1〜3 回目の平均（重み 0 なので単純平均）
    assert out[0].tolist() == [1.0, 0.0]
    assert out[2].tolist() == [1.5, 1.0]  # 時刻 2 は 1 回目の 2 番目、2 回目の 0 番目
    assert out[4][0].item() == pytest.approx(2.0)


def test_ensemble_weights_favor_older_predictions():
    policy = FakePolicy()
    select = ActionSelector(policy, identity, identity, ensemble=1.0, replan=1)
    select({})
    out = select({})[0]  # 1 回目（重み 1）と 2 回目（重み e^-1）
    expected = (1.0 + 2.0 * torch.exp(torch.tensor(-1.0))) / (1 + torch.exp(torch.tensor(-1.0)))
    assert out[0].item() == pytest.approx(expected.item())


def test_reset_clears_state():
    policy = FakePolicy()
    select = ActionSelector(policy, identity, identity, ensemble=0.01, replan=3)
    for _ in range(4):
        select({})
    select.reset()
    assert select.t == 0 and select.chunks == []
    with pytest.raises(ValueError):
        ActionSelector(policy, identity, identity, ensemble=0.01, replan=CHUNK + 1)
