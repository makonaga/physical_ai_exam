"""VLA（SmolVLA）の追加学習の補助の動作確認. Hugging Face Hub には接続しない."""

import pytest

from so101_food.vla.rename_map import DATASET_CAMERAS, build_rename_map

OVERHEAD, WRIST = DATASET_CAMERAS


def test_rename_map_assigns_cameras_in_order():
    expected = ["observation.image", "observation.image2", "observation.image3"]
    assert build_rename_map(expected, [OVERHEAD, WRIST]) == {
        OVERHEAD: "observation.image",
        WRIST: "observation.image2",
    }


def test_rename_map_matches_wrist_by_name():
    expected = ["observation.images.top", "observation.images.side", "observation.images.wrist"]
    mapping = build_rename_map(expected, [OVERHEAD, WRIST])
    assert mapping == {WRIST: "observation.images.wrist", OVERHEAD: "observation.images.top"}


def test_rename_map_is_empty_when_names_already_match():
    assert build_rename_map([OVERHEAD, WRIST, "observation.images.side"], [OVERHEAD, WRIST]) == {}


def test_rename_map_rejects_too_many_cameras():
    with pytest.raises(ValueError):
        build_rename_map(["observation.image"], [OVERHEAD, WRIST])
