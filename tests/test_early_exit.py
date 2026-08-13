"""Phase 12 (M4) — 다단 조기종료 CNN 의 단위 테스트.

검증 포인트(GPU 불필요 — 작은 랜덤 입력으로 규칙만 검증):
    - 양 끝점 수렴: 임계값 > 1 이면 일반 3블록 CNN, 임계값 0 이면 전원 1블록 종료
    - ⭐ **축소 배치가 결과를 바꾸지 않는다** — 조기종료를 '실제 연산 생략'으로 구현할 수 있는
      근거이자, 배치 구성이 달라도 같은 예측이 나온다는 배포 조건
    - 체크포인트 호환: 임계값·종료기록이 state_dict 에 새지 않는다
    - 다중 헤드 손실이 '본 헤드 + 가중치×보조 헤드' 로 계산된다
    - 임계값 선택 규칙이 정확도 조건을 지키며 가장 싼 지점을 고르고, 없으면 조기종료를 끈다
    - tag 축(_aw/_ex)이 기본값에서 생략돼 기존 파일명과 호환된다

무거운 학습은 테스트하지 않는다(train.py --smoke / cascade.py --smoke 로 별도 확인).
"""

import numpy as np
import pytest
import torch

import cnn as cnn_mod
from cascade import select_exit_threshold
from tagging import DEFAULT_AUX_WEIGHT, aux_weight_suffix, build_tag, exit_suffix
from train import build_criterion

NUM_CLASSES = 4


def _model(in_channels: int = 3, seed: int = 0) -> cnn_mod.EarlyExitCNN:
    """재현 가능한 조기종료 모델(평가 모드)."""
    torch.manual_seed(seed)
    net = cnn_mod.build_early_exit_model(NUM_CLASSES, in_channels=in_channels)
    return net.eval()


def _batch(n: int = 8, in_channels: int = 3, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, in_channels, 48, 48, generator=g)


# ── 양 끝점 수렴 ─────────────────────────────────────────────────────────────
def test_no_early_exit_matches_full_depth_final_head():
    # 임계값 > 1 = 아무도 조기종료 안 함 → 전 깊이 최종 헤드와 같아야 한다
    # (= 일반 PayloadCNN 과 같은 계산 경로. 이게 어긋나면 '조기종료의 순수 효과' 비교가 무의미).
    net = _model()
    x = _batch()
    net.exit_threshold = cnn_mod.NO_EARLY_EXIT
    with torch.no_grad():
        got = net(x)
        expected = net.forward_all_heads(x)[-1]
    assert torch.allclose(got, expected, atol=1e-6)
    assert (net.last_exit_stage == net.n_stages - 1).all()


def test_zero_threshold_exits_everyone_at_the_first_block():
    # 임계값 0 = 확신도가 무조건 임계값 이상 → 전원 1블록에서 확정(가장 싼 극단).
    net = _model()
    x = _batch()
    net.exit_threshold = 0.0
    with torch.no_grad():
        got = net(x)
        expected = net.forward_all_heads(x)[0]
    assert (net.last_exit_stage == 0).all()
    assert torch.allclose(got, expected, atol=1e-6)


def test_lower_threshold_never_costs_more_depth():
    # 임계값을 낮출수록 더 일찍 나간다(비용 단조성). 선택 규칙이 이 단조성에 기대고 있다.
    net = _model()
    x = _batch(32)
    depths = []
    for thr in (0.0, 0.3, 0.6, 0.9, cnn_mod.NO_EARLY_EXIT):
        net.exit_threshold = thr
        with torch.no_grad():
            net(x)
        depths.append(float(net.last_exit_stage.float().mean()))
    assert all(a <= b + 1e-9 for a, b in zip(depths, depths[1:]))


# ── ⭐ 축소 배치가 결과를 바꾸지 않는다 ──────────────────────────────────────
def test_shrinking_batch_does_not_change_per_sample_output():
    """조기종료는 확정된 샘플을 배치에서 빼고 남은 것만 다음 블록에 넣는다.

    이 최적화가 정당하려면 '샘플별 결과가 배치 구성과 무관'해야 한다. eval 모드에서 BatchNorm
    이 running stats 를 쓰기 때문에 성립한다. 만약 누군가 train 모드에서 이 경로를 타게 만들면
    배치 통계가 섞여 결과가 흔들리므로, 그 회귀를 여기서 잡는다.
    """
    net = _model()
    net.exit_threshold = 0.4  # 일부만 조기종료되도록(섞인 배치를 만든다)
    x = _batch(16)

    with torch.no_grad():
        batched, stages_batched = net.forward_with_exits(x)
        one_by_one = torch.cat([net.forward_with_exits(x[i:i + 1])[0] for i in range(len(x))])
        stages_single = torch.cat([net.forward_with_exits(x[i:i + 1])[1] for i in range(len(x))])

    assert torch.allclose(batched, one_by_one, atol=1e-6)
    assert torch.equal(stages_batched, stages_single)


def test_every_sample_gets_a_decision():
    # 어떤 임계값에서도 미결정(-1)이 남으면 안 된다 — 남으면 그 샘플은 예측이 0 벡터가 된다.
    net = _model()
    x = _batch(16)
    for thr in (0.0, 0.5, 0.95, cnn_mod.NO_EARLY_EXIT):
        with torch.no_grad():
            _, stages = net.forward_with_exits(x, threshold=thr)
        assert (stages >= 0).all()


def test_empty_batch_is_handled():
    net = _model()
    with torch.no_grad():
        logits, stages = net.forward_with_exits(torch.zeros(0, 3, 48, 48))
    assert logits.shape == (0, NUM_CLASSES)
    assert stages.shape == (0,)


# ── 체크포인트 호환 ──────────────────────────────────────────────────────────
def test_threshold_is_not_part_of_the_checkpoint():
    """임계값은 추론 손잡이지 학습된 값이 아니다.

    state_dict 에 들어가면 임계값만 다른 실행이 체크포인트 호환성을 깨뜨린다(같은 가중치를
    다른 운영점으로 평가하는 것이 M4 의 핵심 비교인데 그게 막힌다).
    """
    net = _model()
    keys = set(net.state_dict().keys())
    assert not any("exit" in k for k in keys)

    net.exit_threshold = 0.5
    other = _model(seed=99)
    other.load_state_dict(net.state_dict())  # 임계값이 달라도 로드가 성립해야 한다
    assert other.exit_threshold == cnn_mod.DEFAULT_EXIT_THRESHOLD


def test_backbone_shape_matches_plain_cnn():
    # 백본이 같아야 "조기종료를 붙였을 때의 순수 효과"만 분리된다. 늘어난 것은 보조 헤드뿐.
    plain = cnn_mod.build_model(NUM_CLASSES, in_channels=3)
    ee = cnn_mod.build_early_exit_model(NUM_CLASSES, in_channels=3)
    n_plain = sum(p.numel() for p in plain.parameters())
    n_ee = sum(p.numel() for p in ee.parameters())
    aux = (32 + 1) * NUM_CLASSES + (64 + 1) * NUM_CLASSES  # 보조 헤드 2개(가중치+편향)
    assert n_ee - n_plain == aux


# ── 종료 분포 ────────────────────────────────────────────────────────────────
def test_exit_distribution_sums_to_one():
    dist = cnn_mod.exit_distribution(np.array([0, 0, 1, 2]), n_stages=3)
    assert dist == pytest.approx([0.5, 0.25, 0.25])
    assert cnn_mod.exit_distribution(np.array([]), n_stages=3) == [0.0, 0.0, 0.0]


# ── 다중 헤드 손실 ───────────────────────────────────────────────────────────
def test_multi_head_loss_is_final_plus_weighted_aux():
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    device = torch.device("cpu")
    target = torch.tensor([0, 1, 2, 3])
    heads = [torch.randn(4, NUM_CLASSES) for _ in range(3)]

    ce = build_criterion(weights, device, aux_weight=0.0)
    weighted = build_criterion(weights, device, aux_weight=0.3)
    # aux_weight=0 이면 본 헤드 손실과 정확히 같다.
    assert ce(heads, target).item() == pytest.approx(ce(heads[-1], target).item(), abs=1e-6)
    # 가중치를 주면 보조 헤드 손실이 그만큼 더해진다.
    expected = (ce(heads[-1], target) + 0.3 * (ce(heads[0], target) + ce(heads[1], target)))
    assert weighted(heads, target).item() == pytest.approx(expected.item(), abs=1e-6)


def test_single_head_model_path_is_unchanged():
    # 단일 헤드 모델은 기존과 완전히 같은 CrossEntropy 여야 한다(경로 분기 없음).
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    crit = build_criterion(weights, torch.device("cpu"), aux_weight=0.3)
    logits = torch.randn(4, NUM_CLASSES)
    target = torch.tensor([0, 1, 2, 3])
    ref = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))(logits, target)
    assert crit(logits, target).item() == pytest.approx(ref.item(), abs=1e-6)


# ── 임계값 선택 규칙 ─────────────────────────────────────────────────────────
def _exit_rows(pairs):
    """(임계값, Macro-F1, 평균깊이) 목록을 스윕 행 형태로."""
    return [{"exit_threshold": t, "macro_f1": m, "accuracy": m,
             "exit_rates": [0.0, 0.0, 1.0], "mean_exit_depth": d}
            for t, m, d in pairs]


def test_select_picks_the_cheapest_point_within_tolerance():
    # 무종료(임계값 최대) Macro-F1=0.9700. 허용 0.0011 안에 드는 것 중 평균 깊이가 가장 얕은 것.
    rows = _exit_rows([(0.0, 0.9000, 1.0), (0.5, 0.9695, 1.4), (0.9, 0.9699, 2.1),
                       (1.01, 0.9700, 3.0)])
    out = select_exit_threshold(rows, tolerance=0.0011)
    assert out["selected"]["exit_threshold"] == 0.5
    assert out["adopted"] is True


def test_select_falls_back_to_no_exit_when_accuracy_never_holds():
    # 정확도 조건을 만족하는 지점이 없으면 조기종료를 끄는 것이 정답이다
    # (docs/08 의 hybrid 처럼 '시도했으나 미채택'이 정직한 결과).
    rows = _exit_rows([(0.0, 0.5000, 1.0), (0.5, 0.9000, 1.5), (1.01, 0.9700, 3.0)])
    out = select_exit_threshold(rows, tolerance=0.0011)
    assert out["selected"]["exit_threshold"] == 1.01
    assert out["adopted"] is False


# ── tag 축 ───────────────────────────────────────────────────────────────────
def test_tag_axes_are_omitted_at_defaults_for_backward_compatibility():
    assert aux_weight_suffix(None) == ""
    assert aux_weight_suffix(DEFAULT_AUX_WEIGHT) == ""
    assert aux_weight_suffix(0.5) == "_aw0.5"
    assert exit_suffix(None) == ""
    assert exit_suffix(0.9) == "_ex0.9"
    # 기존 호출(조기종료 인자 없음)의 파일명이 그대로여야 과거 산출물을 계속 쓸 수 있다.
    assert build_tag("trk", "cnn", "raw", balance=True) == "trk_cnn_raw_bal"
    assert build_tag("trk", "cnn_ee", "raw", balance=True,
                     aux_weight=0.5) == "trk_cnn_ee_raw_aw0.5_bal"
