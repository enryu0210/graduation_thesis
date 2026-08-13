"""Phase 13 신설 평가지표 검증 (docs/13 §1.2)

여기서 고정하는 것은 "값이 그럴듯한가"가 아니라 **정의가 맞는가**다.
손으로 계산할 수 있는 작은 입력을 만들어, 지표가 정의대로 나오는지 못박는다.

왜 이 테스트가 필요한가:
    운영 지점(TPR@FPR)·pAUC·ECE·경보 부하는 전부 "논문 숫자를 그대로 결정하는" 지표다.
    한 번 잘못 구현되면 전 실험이 조용히 틀린 값을 보고하게 되고, 그림이 그럴듯해서
    발견되지 않는다. 그래서 손계산 대조를 남긴다.
"""

import numpy as np
import pytest

import metrics as M


# ── 공통 픽스처 ───────────────────────────────────────────────────────────────
CLASSES = ["Normal", "SQLi", "XSS", "CmdI"]


def _one_hot_scores(labels, confidence=1.0, n_classes=4):
    """지정한 클래스에 confidence 를 주고 나머지에 균등 분배한 확률 행렬을 만든다."""
    scores = np.full((len(labels), n_classes), (1.0 - confidence) / (n_classes - 1))
    scores[np.arange(len(labels)), labels] = confidence
    return scores


# ── attack_score: 다중 클래스 → "공격일 확률" 접기 ─────────────────────────────
def test_attack_score_is_one_minus_normal_probability():
    scores = np.array([[0.9, 0.05, 0.03, 0.02],
                       [0.1, 0.7, 0.1, 0.1]])
    got = M.attack_score(scores, normal_idx=0)
    assert got == pytest.approx([0.1, 0.9])


def test_attack_score_equals_sum_of_attack_columns():
    # 확률 합이 1 이므로 1-P(Normal) 은 공격 열들의 합과 같아야 한다(정의 일치 확인).
    rng = np.random.default_rng(0)
    raw = rng.random((20, 4))
    probs = raw / raw.sum(axis=1, keepdims=True)
    assert M.attack_score(probs, 0) == pytest.approx(probs[:, 1:].sum(axis=1))


# ── TPR@FPR ──────────────────────────────────────────────────────────────────
def test_tpr_at_fpr_returns_an_achievable_operating_point():
    # 정상 100개는 점수 0.0~0.099, 공격 100개는 0.5~0.599 로 완전히 분리된 경우.
    # 어떤 낮은 FPR 에서도 TPR 은 1.0 이어야 한다.
    y = np.array([0] * 100 + [1] * 100)
    score = np.concatenate([np.linspace(0.0, 0.099, 100), np.linspace(0.5, 0.599, 100)])
    out = M.tpr_at_fpr(y, score, 0.001)
    assert out["tpr"] == pytest.approx(1.0)
    assert out["achieved_fpr"] <= 0.001


def test_tpr_at_fpr_never_exceeds_the_target_fpr():
    # 겹치는 분포에서: 목표 FPR 을 초과하는 지점을 고르면 안 된다(보간 금지 규칙).
    rng = np.random.default_rng(42)
    y = np.array([0] * 500 + [1] * 500)
    score = np.concatenate([rng.normal(0.3, 0.1, 500), rng.normal(0.7, 0.1, 500)])
    for target in (0.01, 0.001):
        out = M.tpr_at_fpr(y, score, target)
        assert out["achieved_fpr"] <= target + 1e-12


def test_tpr_at_fpr_is_monotone_in_the_budget():
    # FPR 예산을 늘리면 달성 TPR 이 줄어들 수는 없다.
    rng = np.random.default_rng(1)
    y = np.array([0] * 300 + [1] * 300)
    score = np.concatenate([rng.normal(0.4, 0.15, 300), rng.normal(0.6, 0.15, 300)])
    strict = M.tpr_at_fpr(y, score, 0.001)["tpr"]
    loose = M.tpr_at_fpr(y, score, 0.01)["tpr"]
    assert loose >= strict


def test_tpr_at_fpr_returns_none_when_only_one_class_is_present():
    # 외부 검증셋(Data 2025)처럼 악성만 있는 경우 → ROC 가 정의되지 않으므로 None.
    y = np.ones(10, dtype=int)
    out = M.tpr_at_fpr(y, np.linspace(0, 1, 10), 0.01)
    assert out["tpr"] is None and out["threshold"] is None


# ── pAUC ─────────────────────────────────────────────────────────────────────
def test_pauc_is_one_for_perfectly_separated_scores():
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.zeros(50), np.ones(50)])
    assert M.partial_roc_auc(y, score, 0.01) == pytest.approx(1.0)


def test_pauc_is_half_for_random_scores():
    # McClish 보정으로 [0.5, 1] 로 표준화되므로 무작위 점수는 0.5 근방이어야 한다.
    # (전 구간 ROC-AUC 와 같은 척도가 아님을 이 테스트가 문서화한다.)
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 4000)
    score = rng.random(4000)
    assert M.partial_roc_auc(y, score, 0.01) == pytest.approx(0.5, abs=0.06)


def test_pauc_returns_none_when_roc_is_undefined():
    assert M.partial_roc_auc(np.zeros(10, dtype=int), np.linspace(0, 1, 10)) is None


# ── ECE ──────────────────────────────────────────────────────────────────────
def test_ece_is_zero_when_confidence_matches_accuracy():
    # 확신도 1.0 으로 전부 맞히면 |정확도 - 확신도| = 0 → ECE 0.
    y = np.array([0, 1, 2, 3] * 10)
    scores = _one_hot_scores(y, confidence=1.0)
    cal = M.calibration_metrics(y, scores)
    assert cal["ece"] == pytest.approx(0.0)
    assert cal["mce"] == pytest.approx(0.0)


def test_ece_detects_full_overconfidence():
    # 확신도 1.0 인데 전부 틀리면 gap = |0 - 1| = 1 → ECE 1.0(최악).
    y = np.zeros(20, dtype=int)
    scores = _one_hot_scores(np.ones(20, dtype=int), confidence=1.0)
    cal = M.calibration_metrics(y, scores)
    assert cal["ece"] == pytest.approx(1.0)


def test_ece_matches_hand_computation_on_a_mixed_case():
    # 절반은 확신도 1.0 으로 맞히고(gap 0), 절반은 확신도 1.0 으로 틀린다(gap 1).
    # 두 그룹이 같은 구간에 들어가므로 구간 정확도 0.5, 평균 확신도 1.0 → ECE = 0.5.
    y = np.array([0] * 10 + [1] * 10)
    scores = _one_hot_scores(np.zeros(20, dtype=int), confidence=1.0)
    cal = M.calibration_metrics(y, scores)
    assert cal["ece"] == pytest.approx(0.5)


def test_ece_bins_cover_every_sample():
    # 구간 경계 처리 버그(확신도 1.0 이나 최저 확신도 샘플 누락)를 막는다.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 200)
    raw = rng.random((200, 4))
    scores = raw / raw.sum(axis=1, keepdims=True)
    cal = M.calibration_metrics(y, scores)
    assert sum(b["count"] for b in cal["bins"]) == 200


def test_ece_never_exceeds_mce():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 4, 300)
    raw = rng.random((300, 4))
    scores = raw / raw.sum(axis=1, keepdims=True)
    cal = M.calibration_metrics(y, scores)
    # ECE 는 구간 gap 의 가중평균, MCE 는 그 최대값 → 평균이 최대를 넘을 수 없다.
    assert cal["ece"] <= cal["mce"] + 1e-12


# ── 경보 부하 (base rate) ─────────────────────────────────────────────────────
def test_alert_load_matches_hand_computation():
    # docstring 의 예시를 그대로 검증: FPR=0.002, TPR=0.99, pi=0.001, N=100만
    rows = M.alert_load(tpr=0.99, fpr=0.002, prevalences=(0.001,), per_requests=1_000_000)
    assert len(rows) == 1
    row = rows[0]
    assert row["true_alerts"] == pytest.approx(990.0)
    assert row["false_alerts"] == pytest.approx(1998.0)
    assert row["precision_at_prevalence"] == pytest.approx(990 / 2988)


def test_alert_load_precision_drops_as_attacks_get_rarer():
    # base rate fallacy 의 핵심: 같은 모델인데 공격이 드물어지면 정밀도가 무너진다.
    rows = M.alert_load(tpr=0.99, fpr=0.01, prevalences=(0.1, 0.01, 0.001))
    precisions = [r["precision_at_prevalence"] for r in rows]
    assert precisions == sorted(precisions, reverse=True)


def test_alert_load_is_empty_when_rates_are_unavailable():
    assert M.alert_load(None, 0.01) == []
    assert M.alert_load(0.9, None) == []


# ── compute_metrics 통합 ──────────────────────────────────────────────────────
def test_compute_metrics_no_longer_reports_mcc():
    # Phase 13: 교수 지시로 MCC 를 완전히 제거했다(docs/13 §1.1.1).
    # 이 테스트는 "되살아나면 실패"하도록 남긴다 — 문서와 코드가 어긋나는 것을 막는다.
    y = np.array([0, 1, 2, 3, 0, 1])
    result = M.compute_metrics(y, y, CLASSES, y_score=_one_hot_scores(y))
    assert "mcc" not in result


def test_compute_metrics_includes_the_new_phase13_blocks():
    y = np.array([0, 1, 2, 3] * 5)
    result = M.compute_metrics(y, y, CLASSES, y_score=_one_hot_scores(y, 0.9))

    assert result["calibration"]["ece"] is not None
    op = result["operating_points"]
    assert op["pauc"] is not None
    # 목표 FPR 개수만큼 운영 지점이 나와야 한다.
    assert len(op["tpr_at_fpr"]) == len(M.TARGET_FPRS)
    # 경보 부하는 가정 공격 비율 개수만큼.
    assert len(result["attack_focused"]["alert_load"]) == len(M.DEPLOYMENT_ATTACK_PREVALENCES)


def test_operating_points_are_skipped_without_scores():
    # 확률을 못 주는 모델(일부 sklearn 설정)에서도 평가가 죽지 않아야 한다.
    y = np.array([0, 1, 2, 3])
    result = M.compute_metrics(y, y, CLASSES)
    assert "operating_points" not in result
    assert "calibration" not in result
    # 예측만으로 계산되는 보안 지표는 그대로 나와야 한다.
    assert result["attack_focused"]["benign_evasion_rate"] == pytest.approx(0.0)


def test_observed_prevalence_is_reported_for_base_rate_contrast():
    # 공격 3 : 정상 1 → 관측 공격 비율 0.75. 이 값이 배포 가정(1%)과 얼마나 다른지가
    # base rate fallacy 의 크기이므로 반드시 함께 보고돼야 한다.
    y = np.array([0, 1, 2, 3])
    result = M.compute_metrics(y, y, CLASSES)
    assert result["attack_focused"]["observed_attack_prevalence"] == pytest.approx(0.75)


def test_format_summary_shows_new_metrics_and_not_mcc():
    y = np.array([0, 1, 2, 3] * 5)
    result = M.compute_metrics(y, y, CLASSES, y_score=_one_hot_scores(y, 0.9))
    line = M.format_summary("cnn", result)
    assert "MCC" not in line
    assert "ECE=" in line
    assert "pAUC=" in line
    assert "TPR@" in line


# ── 그림 저장 ─────────────────────────────────────────────────────────────────
def test_reliability_diagram_is_written(tmp_path):
    y = np.array([0, 1, 2, 3] * 10)
    cal = M.calibration_metrics(y, _one_hot_scores(y, 0.8))
    out = M.save_reliability_diagram(cal, tmp_path / "cal.png")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_reliability_diagram_skips_empty_calibration(tmp_path):
    # 빈 구간만 있는 입력에서 그림을 만들려 하면 예외 없이 None 을 돌려줘야 한다.
    empty = {"ece": 0.0, "mce": 0.0, "n_bins": 2,
             "bins": [{"lo": 0.0, "hi": 0.5, "count": 0, "avg_confidence": None,
                       "accuracy": None, "gap": None}]}
    assert M.save_reliability_diagram(empty, tmp_path / "none.png") is None
