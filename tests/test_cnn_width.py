"""T4b 채널 폭 축 — ×1 은 기존 체크포인트·파일명과 호환되고, 다른 폭은 tag 로 갈린다."""
import pytest
import torch

import cnn as cnn_mod
from cascade import checkpoint_tag
from tagging import build_tag


def test_width_one_keeps_existing_state_dict_shapes():
    """폭 1.0 이 기존 32/64/128 구조와 같아야 기존 .pt 를 그대로 읽는다."""
    shapes = {k: v.shape for k, v in cnn_mod.build_model(4, in_channels=3).state_dict().items()}
    assert shapes["features.2.0.weight"] == (128, 64, 3, 3)
    assert shapes["classifier.weight"] == (4, 128)


@pytest.mark.parametrize("width,gap", [(0.5, 64), (2.0, 256)])
def test_width_scales_channels_and_runs(width, gap):
    net = cnn_mod.build_model(4, in_channels=3, width=width).eval()
    assert net.classifier.in_features == gap
    assert net(torch.zeros(2, 3, 48, 48)).shape == (2, 4)


def test_width_tag_axis():
    assert build_tag("srbh_4class", "cnn", channels="rgb", width=1.0) == build_tag("srbh_4class", "cnn", channels="rgb")
    assert build_tag("srbh_4class", "cnn", channels="rgb", width=0.5).endswith("_w0.5")
    # 텍스트 2차에는 폭 축을 붙이지 않는다(엉뚱한 체크포인트 이름 방지).
    assert "_w" not in checkpoint_tag("srbh_4class", "charcnn", "raw", True, width=2.0)
    assert checkpoint_tag("srbh_4class", "cnn", "raw", False, "rgb", None, width=2.0).endswith("_rgb_w2")


def test_non_positive_width_rejected():
    with pytest.raises(ValueError):
        cnn_mod.build_model(4, width=0)
