"""Phase 9 하이브리드 캐스케이드의 순수 로직 단위 테스트.

검증 포인트(체크포인트·GPU 불필요 — 확률 행렬을 직접 만들어 규칙만 검증):
    - 캐스케이드가 양 끝에서 두 단독 모델로 정확히 수렴하는가(τ=0 → 1차, τ>1 → 2차)
    - 확신도 임계값이 '미만(<)' 규칙대로 정확한 샘플만 2차로 넘기는가
    - τ 격자가 에스컬레이션 0%~100% 양 끝점을 포함하는가
    - τ 선택 규칙(match-teacher / budget)이 의도한 지점을 고르는가
    - oracle 라우팅이 '1차가 틀린 샘플만' 넘기는 상한을 계산하는가

무거운 학습·추론은 테스트하지 않는다(cascade.py --smoke 로 별도 확인).
"""

import numpy as np
import pytest

from cascade import cascade_apply, make_tau_grid, oracle_point, select_tau


# 1차 확률: 0·1번 샘플은 확신(0.9), 2·3번 샘플은 애매(0.5). 예측은 각각 0,1,0,1.
P1 = np.array([
    [0.90, 0.10],
    [0.10, 0.90],
    [0.50, 0.50 - 1e-9],
    [0.50 - 1e-9, 0.50],
])
# 2차 확률: 1차와 정반대로 예측하도록 구성 → 어떤 샘플이 넘어갔는지 예측만 보고 알 수 있다.
P2 = np.array([
    [0.20, 0.80],
    [0.80, 0.20],
    [0.20, 0.80],
    [0.80, 0.20],
])


# ── cascade_apply: 게이트 규칙 ────────────────────────────────────────────────
def test_tau_zero_is_stage1_alone():
    # τ=0 이면 확신도가 0 미만인 샘플이 없으므로 아무도 넘어가지 않는다(= 제안 CNN 단독).
    pred, _, escalate = cascade_apply(P1, P2, 0.0)
    assert not escalate.any()
    assert np.array_equal(pred, P1.argmax(axis=1))


def test_tau_above_one_is_stage2_alone():
    # τ>1 이면 모든 확신도가 임계값 미만이라 전부 넘어간다(= char-CNN 단독).
    pred, _, escalate = cascade_apply(P1, P2, 1.01)
    assert escalate.all()
    assert np.array_equal(pred, P2.argmax(axis=1))


def test_only_low_confidence_samples_escalate():
    # τ=0.7 이면 확신도 0.9 인 앞의 둘은 1차 유지, 0.5 인 뒤의 둘만 2차로 넘어간다.
    pred, _, escalate = cascade_apply(P1, P2, 0.7)
    assert escalate.tolist() == [False, False, True, True]
    assert pred[:2].tolist() == P1.argmax(axis=1)[:2].tolist()
    assert pred[2:].tolist() == P2.argmax(axis=1)[2:].tolist()


def test_combined_probs_come_from_the_deciding_stage():
    # 결합 확률(점수 기반 지표용)도 '실제로 판정한 단계'의 것이어야 한다.
    _, probs, _ = cascade_apply(P1, P2, 0.7)
    assert np.allclose(probs[0], P1[0])
    assert np.allclose(probs[3], P2[3])


# ── make_tau_grid: 곡선의 양 끝점 보장 ────────────────────────────────────────
def test_tau_grid_spans_both_extremes():
    conf = np.array([0.5, 0.6, 0.9, 0.99])
    grid = make_tau_grid(conf, n=5)
    assert grid[0] == 0.0          # 에스컬레이션 0%(1차 단독)
    assert grid[-1] > conf.max()   # 에스컬레이션 100%(2차 단독)
    assert np.all(np.diff(grid) > 0)  # 정렬·중복 제거 상태


# ── select_tau: 운영점 선택 규칙 ──────────────────────────────────────────────
def _rows(pairs):
    """(escalation, macro_f1) 목록을 스윕 행 형태로 만든다."""
    return [{"tau": 0.1 * i, "escalation_rate": e, "macro_f1": f, "mcc": f,
             "accuracy": f, "benign_evasion_rate": 0.0,
             "normal_false_positive_rate": 0.0}
            for i, (e, f) in enumerate(pairs)]


def test_select_match_teacher_picks_cheapest_point_reaching_teacher():
    # 0.95 를 처음 넘는 지점이 둘(0.3, 0.8)이면 더 싼 0.3 을 골라야 한다.
    rows = _rows([(0.0, 0.90), (0.3, 0.96), (0.8, 0.97), (1.0, 0.95)])
    best = select_tau(rows, "match-teacher", 0.1, teacher_macro_f1=0.95)
    assert best["escalation_rate"] == 0.3


def test_select_match_teacher_falls_back_to_best_when_unreachable():
    # 아무도 교사 성능에 못 미치면 val Macro-F1 최대 지점으로 대체하고 그 사실을 note 에 남긴다.
    rows = _rows([(0.0, 0.90), (0.5, 0.93), (1.0, 0.92)])
    best = select_tau(rows, "match-teacher", 0.1, teacher_macro_f1=0.99)
    assert best["escalation_rate"] == 0.5
    assert "대체" in best["note"]


def test_select_budget_respects_the_escalation_cap():
    # 예산 0.4 를 넘는 (0.8, 0.99) 지점은 아무리 좋아도 선택되면 안 된다.
    rows = _rows([(0.0, 0.90), (0.3, 0.94), (0.8, 0.99)])
    best = select_tau(rows, "budget", 0.4, teacher_macro_f1=0.99)
    assert best["escalation_rate"] == 0.3


def test_select_budget_never_returns_empty_when_cap_is_zero():
    # 예산이 0 이면 실행 가능한 지점이 없을 수 있다 → 가장 싼 지점으로 안전하게 되돌린다.
    rows = _rows([(0.1, 0.90), (0.5, 0.95)])
    best = select_tau(rows, "budget", 0.0, teacher_macro_f1=0.99)
    assert best["escalation_rate"] == 0.1


# ── oracle_point: 라우팅 상한 ─────────────────────────────────────────────────
def test_oracle_escalates_exactly_the_stage1_errors():
    # 정답이 [0,1,1,1] 이면 1차 예측 [0,1,0,1] 중 2번 샘플만 틀렸다 → 에스컬레이션 1/4.
    y = np.array([0, 1, 1, 1])
    result = oracle_point(P1, P2, y, ["a", "b"])
    assert result["escalation_rate"] == pytest.approx(0.25)
    # 그 샘플을 2차(예측 1)가 맞히므로 상한은 완전 정확이 된다.
    assert result["macro_f1"] == pytest.approx(1.0)
