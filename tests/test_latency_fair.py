"""E1 지연 측정의 입력이 학습 때 입력과 같아야 한다 — 다르면 '다른 모델'의 지연을 재게 된다."""
import numpy as np

import data_image
import data_text
from payload_to_image import payload_to_rgb_image
from tagging import build_tag

from src.eval import latency_fair

TEXTS = ["/index.php?id=1' OR '1'='1", "", "/a", "한글 경로/" + "x" * 3000]


def test_prep_img_matches_training_transform():
    images = np.stack([payload_to_rgb_image(text, side=48) for text in TEXTS])
    expected = data_image.make_torch_dataset(images, np.zeros(len(TEXTS), dtype=np.int64)).tensors[0]
    for i, text in enumerate(TEXTS):
        assert np.array_equal(latency_fair.prep_img(text)[0].numpy(), expected[i].numpy())


def test_real_length_encoding_is_prefix_of_padded():
    for text in TEXTS:
        padded = data_text.encode_byte_sequence(text, latency_fair.MAX_LEN)
        assert np.array_equal(latency_fair.enc_seq(text, True)[0].numpy(), padded)
        real = latency_fair.enc_seq(text, False)[0].numpy()
        n_bytes = min(len(text.encode("utf-8")), latency_fair.MAX_LEN)
        assert len(real) == max(n_bytes, latency_fair.MIN_REAL_LEN)
        assert np.array_equal(real, padded[: len(real)])


def test_judgment_fields_follow_prereg_formulas():
    summary = {name: {"p50": 1.0, "p95": 1.0, "p99": 1.0, "mean": 1.0}
               for name in (*latency_fair.COMPONENTS, *latency_fair.COMPOSITES)}
    summary["stage1_e2e"] = dict.fromkeys(("p50", "p95", "p99", "mean"), 2.0)
    summary["stage2_real_e2e"] = dict.fromkeys(("p50", "p95", "p99", "mean"), 4.0)
    got = latency_fair.judge(summary, sample_rate=0.5, e_pi=0.25)
    assert got["e1a_withdraw"] is True                      # 2차 순전파(1.0) ≤ 1차 전체(2.0) → 철회 조건
    assert got["upper_bound_real"] == 2.0
    assert got["speedup_formula"]["fair_e2e_real_p50"]["sample_rate"] == 4.0 / (2.0 + 0.5 * 4.0)
    assert got["speedup_formula"]["fair_e2e_real_p50"]["pi_0.033"] == 4.0 / (2.0 + 0.25 * 4.0)


def test_cv_seed_axis_keeps_existing_names():
    base = build_tag("srbh_4class", "cnn", channels="rgb", sealed_test=True)
    assert base == "srbh_4class_cnn_raw_rgb_sealed"
    assert build_tag("srbh_4class", "cnn", channels="rgb", sealed_test=True, cv_seed=42) == base
    assert build_tag("srbh_4class", "cnn", channels="rgb", sealed_test=True, cv_seed=43) == base + "_s43"
    assert build_tag("srbh_4class", "charcnn", cv_seed=43) == "srbh_4class_charcnn_raw_s43"


def test_charcnn_parts_matches_full_forward():
    # 분해 순전파가 원 모델과 다른 연산을 하면 단계별 시간이 다른 모델의 시간이 된다(E7).
    import torch
    from text_models import CharCNN

    torch.manual_seed(0)
    net = CharCNN(num_classes=4).eval()
    aux = torch.nn.Linear(128, 4).eval()
    with torch.inference_mode():
        for text in TEXTS:
            x = latency_fair.enc_seq(text, False)
            logits, parts = latency_fair.charcnn_parts(net, aux, x)
            assert torch.allclose(logits, net(x), atol=1e-6)
            assert len(parts) == 6 and min(parts) >= 0


def test_breakdown_gates_follow_prereg_formulas():
    p50 = {"enc": 0.01, "emb": 0.02, "conv3": 0.05, "conv5": 0.1, "conv7": 0.15, "head": 0.02,
           "aux3": 0.01, "floor": 0.03, "full": 0.34}
    got = latency_fair.breakdown_gates(p50)
    t2 = 0.35
    assert abs(got["dl_b"]["speedup_ub"] - t2 / (0.04 + 0.053 * t2)) < 1e-12
    assert abs(got["dl_c"]["speedup_ub"] - t2 / (0.09 + 0.053 * 0.27)) < 1e-12
    assert got["dl_b"]["pass"] and got["dl_c"]["pass"]
