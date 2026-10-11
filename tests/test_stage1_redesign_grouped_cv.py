"""E2(1차 재설계 후보)와 T5(템플릿 묶음 분할)의 최소 점검."""
import numpy as np
import torch

import cnn
from tagging import build_tag

from src.eval import cross_validate as CV


def test_e2_variants_keep_input_and_output_contract():
    x = torch.randn(2, 3, 48, 48)
    for name in cnn.E2_VARIANTS:
        model = cnn.build_e2_variant(name, 4, in_channels=3).eval()
        assert model(x).shape == (2, 4), name
    # 자르기 후보는 9행 아래를 보지 않는다 — 아래쪽을 바꿔도 출력이 같아야 한다.
    crop = cnn.build_e2_variant("cnn_crop8", 4).eval()
    y = x.clone()
    y[:, :, 8:] = 0
    assert torch.equal(crop(x), crop(y))


def test_template_groups_merge_requests_that_differ_only_in_values():
    texts = ["/item?id=12&sid=0123456789abcdef0123", "/item?id=987&sid=fedcba98765432100000",
             "/item?name=bob", "/item?id=1 OR 1=1"]
    groups = CV.template_groups(texts)
    assert groups[0] == groups[1]
    assert len(set(groups.tolist())) == 3


def test_inner_group_split_keeps_templates_apart():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 200, size=2000)
    y = groups % 4                                   # 묶음 안에서 라벨이 같게
    train, val = CV.inner_group_split(y, groups, 0.1, seed=1)
    assert not set(groups[train]) & set(groups[val])
    assert 0.05 < len(val) / len(y) < 0.15


def test_grouped_cv_gets_its_own_tag():
    base = build_tag("srbh_4class", "cnn", channels="rgb", sealed_test=True)
    assert build_tag("srbh_4class", "cnn", channels="rgb", sealed_test=True, group_template=True) == base + "_gtpl"
    assert build_tag("srbh_4class", "cnn_gmp", channels="rgb", sealed_test=True) == "srbh_4class_cnn_gmp_raw_rgb_sealed"
