"""Phase 12 (M1) — 표현 불일치 회피 탐지의 순수 로직 단위 테스트.

검증 포인트(체크포인트·GPU 불필요 — 확률 행렬을 직접 만들어 규칙만 검증):
    - 세 점수함수가 '불일치가 클수록 점수가 크다'는 방향을 지키는가
    - 모드 A 의 판정 가능 집합이 **캐스케이드가 실제로 2차를 돌린 집합과 정확히 같은가**
      (이게 어긋나면 "추가 비용 0" 주장 자체가 거짓이 된다)
    - 임계값이 clean 분포에서 목표 오탐률대로 뽑히는가, 근거가 없으면 안전하게 침묵하는가
    - 상보성 집계가 detection/correct 두 기준을 각각 옳게 세는가
    - 판정(H5-1/H5-2)이 docs/11 §7 의 경계값을 그대로 따르는가

무거운 추론은 테스트하지 않는다(run_tamper.py --smoke 로 별도 확인).
"""

import numpy as np
import pytest

import tamper as T
from cascade import cascade_apply


# 1차 확률: 0·1번은 확신(0.9), 2·3번은 애매(0.5).
P1 = np.array([
    [0.90, 0.10],
    [0.10, 0.90],
    [0.50, 0.50 - 1e-9],
    [0.50 - 1e-9, 0.50],
])
# 2차 확률: 1차와 정반대 예측.
P2 = np.array([
    [0.20, 0.80],
    [0.80, 0.20],
    [0.20, 0.80],
    [0.80, 0.20],
])


# ── 점수함수 ─────────────────────────────────────────────────────────────────
def test_identical_distributions_score_zero():
    # 두 표현이 완전히 같게 보면 불일치가 없어야 한다(clean 에서의 기대 동작).
    assert np.allclose(T.tampering_scores(P1, P1, "js_divergence"), 0.0)
    assert np.allclose(T.tampering_scores(P1, P1, "top1_disagree"), 0.0)
    assert np.allclose(T.tampering_scores(P1, P1, "conf_gap"), 0.0)


def test_js_divergence_is_bounded_and_symmetric():
    a = T.tampering_scores(P1, P2, "js_divergence")
    b = T.tampering_scores(P2, P1, "js_divergence")
    assert np.allclose(a, b)              # 대칭 — 어느 쪽이 '정답'이라 가정하지 않는다
    assert np.all((a >= 0.0) & (a <= 1.0))  # 밑 2 JSD 는 [0,1] 유계
    assert a[0] > 0.0                     # 서로 반대로 판정하면 값이 커진다


def test_top1_disagree_is_binary_and_matches_argmax():
    s = T.tampering_scores(P1, P2, "top1_disagree")
    assert set(np.unique(s)) <= {0.0, 1.0}
    assert np.array_equal(s.astype(bool), P1.argmax(axis=1) != P2.argmax(axis=1))


def test_conf_gap_is_positive_when_stage1_wavers():
    # 회피 변형의 전형적 서명: 1차 확신도는 무너지고(0.5) 2차는 버틴다(0.8).
    s = T.tampering_scores(P1, P2, "conf_gap")
    assert s[2] > s[0]  # 2번(1차 애매) > 0번(1차 확신)
    assert s[2] == pytest.approx(0.8 - 0.5, abs=1e-6)


def test_unknown_scorer_is_rejected():
    with pytest.raises(ValueError):
        T.tampering_scores(P1, P2, "cosine")


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        T.tampering_scores(P1, P2[:2], "js_divergence")


# ── 모드 A: 캐스케이드와 같은 집합이어야 한다 ────────────────────────────────
@pytest.mark.parametrize("tau", [0.0, 0.55, 0.7, 1.01])
def test_mode_a_population_equals_actual_stage2_calls(tau):
    """모드 A 의 판정 가능 집합 = 캐스케이드가 실제로 2차를 호출한 집합.

    두 규칙이 조금이라도 어긋나면, 돌리지도 않은 2차의 확률로 점수를 내게 된다
    → M1 의 유일한 기여인 '추가 비용 정확히 0' 이 성립하지 않는다.
    """
    _, _, escalate = cascade_apply(P1, P2, tau)
    assert np.array_equal(T.escalated_mask(P1, tau), escalate)


# ── 임계값 ───────────────────────────────────────────────────────────────────
def test_threshold_hits_the_target_false_positive_rate():
    rng = np.random.default_rng(0)
    clean = rng.uniform(0, 1, size=10_000)
    thr = T.select_threshold(clean, target_fpr=0.01)
    # 초과 규칙(>)이므로 clean 중 임계값을 넘는 비율이 목표치 근처여야 한다.
    assert T.flag(clean, thr).mean() == pytest.approx(0.01, abs=0.003)


def test_threshold_without_clean_samples_flags_nothing():
    # 판정 가능한 clean 샘플이 없으면 임계값을 정할 근거가 없다.
    # 조용히 전부 탐지하는 것보다 아무것도 탐지하지 않는 쪽이 안전하다.
    thr = T.select_threshold(np.array([]), target_fpr=0.01)
    assert thr == float("inf")
    assert not T.flag(np.array([0.9, 1.0]), thr).any()


def test_binary_threshold_flags_only_disagreements():
    # top1_disagree 처럼 값이 0/1 뿐이어도, clean 불일치율이 목표 FPR 보다 낮으면
    # 임계값 0.0 이 잡혀 '불일치한 것만' 플래그된다(분위수가 뭉개지지 않는다).
    clean = np.zeros(1000)
    thr = T.select_threshold(clean, target_fpr=0.01)
    assert T.flag(np.array([0.0, 1.0]), thr).tolist() == [False, True]


# ── 이진 판별 ────────────────────────────────────────────────────────────────
def test_binary_eval_perfect_separation():
    res = T.binary_eval(np.zeros(50), np.ones(50), threshold=0.5)
    assert res["roc_auc"] == pytest.approx(1.0)
    assert res["tpr"] == pytest.approx(1.0)
    assert res["fpr"] == pytest.approx(0.0)


def test_binary_eval_constant_scores_are_uninformative_not_an_error():
    # 전부 같은 점수면 AUC 가 수학적으로 정의되지 않지만, 실험 결과로는 '판별력 없음'이다.
    # 예외로 죽이지 않고 0.5 로 남겨야 실행 전체가 멈추지 않는다.
    res = T.binary_eval(np.zeros(10), np.zeros(10), threshold=0.5)
    assert res["roc_auc"] == pytest.approx(0.5)


def test_binary_eval_without_positives_returns_none_auc():
    res = T.binary_eval(np.zeros(10), np.array([]), threshold=0.5)
    assert res["roc_auc"] is None
    assert res["n_positive"] == 0


# ── 상보성(G3) ───────────────────────────────────────────────────────────────
def test_complementarity_counts_detection_and_correct_separately():
    # 클래스: 0=Normal, 1=SQLi, 2=XSS. 정상 샘플은 집계에서 빠져야 한다.
    y = np.array([0, 1, 1, 2])
    pred_a = np.array([0, 1, 0, 1])  # 1번 정답 / 2번 미탐(Normal) / 3번 오귀속
    pred_b = np.array([0, 2, 1, 2])  # 1번 오귀속 / 2번 탐지+정답 / 3번 정답
    out = T.complementarity(y, pred_a, pred_b, normal_idx=0)

    assert out["n_attacks"] == 3
    # detection: a 는 2번을 Normal 로 흘렸고, b 는 셋 다 공격으로 잡았다.
    assert out["detection"] == {"both": 2, "a_only": 0, "b_only": 1, "neither": 0,
                                "a_rate": pytest.approx(2 / 3),
                                "b_rate": pytest.approx(1.0)}
    # correct: a 는 1번만, b 는 2·3번만 맞혔다 → 서로 배타적(상보적)
    assert out["correct"]["both"] == 0
    assert out["correct"]["a_only"] == 1
    assert out["correct"]["b_only"] == 2


def test_complementarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        T.complementarity(np.array([0, 1]), np.array([0, 1]), np.array([0]), normal_idx=0)


# ── 판정 기준(docs/11 §7) ────────────────────────────────────────────────────
@pytest.mark.parametrize("auc,expected", [
    (0.95, "효과 있음"),
    (0.90, "효과 있음"),     # 경계값은 통과(기준이 '≥ 0.90')
    (0.80, "보조 지표로만 보고"),
    (0.74, "기각"),
    (None, "판정 불가(양성/음성 표본 부족)"),
])
def test_h51_follows_the_prefixed_thresholds(auc, expected):
    assert T.verdict(auc, 0.005, 0.5)["H5-1"]["result"] == expected


def test_h52_needs_both_low_false_positives_and_enough_coverage():
    assert T.verdict(0.95, 0.005, 0.35)["H5-2"]["result"] == "충족"
    assert T.verdict(0.95, 0.02, 0.35)["H5-2"]["result"] == "미충족"   # 오탐 초과
    assert T.verdict(0.95, 0.005, 0.10)["H5-2"]["result"] == "미충족"  # 커버리지 부족
