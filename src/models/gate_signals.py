"""확률만으로 교정과 게이트 선택을 평가하여 GPU 실측과 CPU 분석을 분리한다."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import spearmanr

from src.eval.metrics import calibration_metrics, compute_metrics


def _probabilities(probs):
    """잘못된 확률을 자동 정규화하면 입력 오류를 숨기므로 거부한다."""
    values = np.asarray(probs)
    if (values.ndim != 2 or values.shape[1] < 2
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError("확률은 행 합이 1인 유한한 N×K 배열이어야 합니다(K>=2).")
    return values


def _labels(y, n, k):
    """라벨을 정수로 강제 변환하지 않아 소수 라벨의 조용한 절삭을 막는다."""
    labels = np.asarray(y)
    if (labels.shape != (n,) or not n or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any() or (labels >= k).any()):
        raise ValueError("정답은 표본 수와 일치하는 비어 있지 않은 클래스 정수 배열이어야 합니다.")
    return labels


def _confidence(conf, n=None):
    """빈 평가 집합과 비유한 순위가 예산 계산에 섞이지 않도록 한다."""
    values = np.asarray(conf)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.isfinite(values).all()
            or (n is not None and len(values) != n)):
        raise ValueError("확신도는 표본 수와 일치하는 비어 있지 않은 유한한 1차원 배열이어야 합니다.")
    return values


def apply_temperature(probs, T) -> np.ndarray:
    """로그 공간에서 정규화하여 작은 확률과 낮은 온도에서도 오버플로를 피한다."""
    probs = _probabilities(probs)
    if T is None or not np.isscalar(T) or not np.isfinite(T) or T <= 0:
        raise ValueError("온도는 유한한 양수여야 합니다.")
    logits = np.log(np.clip(probs.astype(float), 1e-12, 1)) / T
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_temperature(probs_val, y_val, bounds=(0.05, 20.0)) -> float:
    """양의 온도 전체를 균형 있게 탐색하도록 log T 공간에서 val NLL을 최소화한다."""
    probs = _probabilities(probs_val)
    labels = _labels(y_val, len(probs), probs.shape[1])
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 < limits[0] < limits[1]:
        raise ValueError("온도 탐색 범위는 증가하는 유한한 양수 두 개여야 합니다.")
    logits = np.log(np.clip(probs.astype(float), 1e-12, 1))

    def nll(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    result = minimize_scalar(nll, bounds=tuple(np.log(limits)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("온도 최적화가 수렴하지 않았습니다.")
    return float(np.exp(result.x))


def gate_confidence(probs, signal, T=None) -> np.ndarray:
    """모든 신호의 방향을 확신도 증가로 맞춰 같은 미만 게이트를 사용한다."""
    probs = _probabilities(probs)
    if signal == "msp":
        return probs.max(axis=1)
    if signal not in {"msp_T", "margin", "entropy"}:
        raise ValueError(f"알 수 없는 게이트 신호: {signal}")
    calibrated = apply_temperature(probs, T)
    if signal == "msp_T":
        return calibrated.max(axis=1)
    if signal == "margin":
        top = np.sort(calibrated, axis=1)[:, -2:]
        return top[:, 1] - top[:, 0]
    # 언더플로로 0이 생겨도 0*log(0)을 정확히 0으로 처리한다.
    logs = np.zeros_like(calibrated)
    np.log(calibrated, out=logs, where=calibrated > 0)
    return np.sum(calibrated * logs, axis=1)


def tau_for_budget(conf_val, budget) -> float:
    """동률 묶음을 쪼갤 수 없는 미만 게이트에서 예산 이내 최대 승급 수를 고른다."""
    conf = _confidence(conf_val)
    if not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("예산은 0 이상 1 이하여야 합니다.")
    values, counts = np.unique(conf, return_counts=True)
    before = np.r_[0, np.cumsum(counts)]
    allowed = np.flatnonzero(before / len(conf) <= budget)[-1]
    # 전량 승급은 최댓값보다 한 표현 단위 큰 수로 나타낸다.
    if allowed < len(values):
        return float(values[allowed])
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype(float)
    return float(np.nextafter(dtype.type(values[-1]), dtype.type(np.inf)))


def cascade_apply_signal(p1, p2, conf, tau):
    """확률 dtype과 선택 연산을 유지하여 기존 MSP 캐스케이드와 비트 단위로 일치시킨다."""
    p1, p2 = _probabilities(p1), _probabilities(p2)
    if p1.shape != p2.shape or not np.isscalar(tau) or np.isnan(tau):
        raise ValueError("두 확률 행렬의 모양과 임계값을 확인하세요.")
    conf = _confidence(conf, len(p1))
    escalate = conf < tau
    probs = np.where(escalate[:, None], p2, p1)
    return probs.argmax(axis=1), probs, escalate


def budget_curve(p1, p2, y_true, conf, n_classes, rates) -> list[dict]:
    """안정 정렬로 동률 순서를 고정해 신호 간 동일한 승급 예산을 비교한다."""
    p1, p2 = _probabilities(p1), _probabilities(p2)
    if p1.shape != p2.shape or n_classes != p1.shape[1]:
        raise ValueError("확률 모양과 클래스 수가 일치해야 합니다.")
    truth = _labels(y_true, len(p1), n_classes)
    order = np.argsort(_confidence(conf, len(p1)), kind="stable")
    first, second = p1.argmax(axis=1), p2.argmax(axis=1)
    rows = []
    for rate in rates:
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("승급 비율은 0 이상 1 이하여야 합니다.")
        k = round(float(rate) * len(p1))
        pred = first.copy()
        pred[order[:k]] = second[order[:k]]
        score = compute_metrics(truth, pred, [str(i) for i in range(n_classes)])["macro_f1"]
        rows.append({"rate": float(rate), "k": k, "escalation_rate": k / len(p1), "macro_f1": score})
    return rows


def over_escalation_ratio(p1, p2, y_true, conf, n_classes, target_macro_f1, step=0.005) -> dict:
    """도달 실패와 오라클 분모 0을 수치 0으로 위장하지 않고 None으로 남긴다."""
    if not np.isfinite(step) or not 0 < step <= 1 or not np.isfinite(target_macro_f1):
        raise ValueError("격자 간격은 (0,1]이고 목표는 유한해야 합니다.")
    rates = np.r_[np.arange(0, 1, step), 1.0]
    curve = budget_curve(p1, p2, y_true, conf, n_classes, rates)
    reached = [row["escalation_rate"] for row in curve if row["macro_f1"] >= target_macro_f1]
    needed = min(reached) if reached else None
    oracle = float(np.mean(np.asarray(p1).argmax(axis=1) != np.asarray(y_true)))
    return {"needed_escalation": needed, "oracle_escalation": oracle,
            "ratio": needed / oracle if needed is not None and oracle else None,
            "reached": needed is not None, "step": float(step)}


def signal_report(p1_val, y_val, p1_test, p2_test, y_test, n_classes,
                  target_macro_f1, budgets=(0.05, 0.10, 0.30)) -> dict:
    """온도와 임계값은 val에서만 고정하고 test에는 그대로 이식한다."""
    temperature = fit_temperature(p1_val, y_val)
    p1_test = _probabilities(p1_test)
    if np.asarray(p1_val).shape[1] != p1_test.shape[1]:
        raise ValueError("val과 test 클래스 수가 다릅니다.")
    truth = _labels(y_test, len(p1_test), n_classes)
    calibrated = apply_temperature(p1_test, temperature)
    calibration = {name: {key: value[key] for key in ("ece", "mce", "brier")}
                   for name, value in (("before", calibration_metrics(truth, p1_test)),
                                       ("after", calibration_metrics(truth, calibrated)))}
    reference = gate_confidence(p1_test, "msp")
    signals = {}
    budgets = tuple(budgets)
    for signal in ("msp", "msp_T", "margin", "entropy"):
        conf = gate_confidence(p1_test, signal, temperature)
        conf_val = gate_confidence(p1_val, signal, temperature)
        # 상수 신호에서는 순위상관 자체가 정의되지 않아 JSON null로 기록한다.
        correlation = None
        if len(conf) > 1 and np.ptp(conf) > 0 and np.ptp(reference) > 0:
            correlation = float(spearmanr(reference, conf).statistic)
        transfer = []
        for budget in budgets:
            tau = tau_for_budget(conf_val, budget)
            transfer.append({"budget": float(budget), "tau": tau,
                             "val_escalation_rate": float(np.mean(conf_val < tau)),
                             "escalation_rate": float(np.mean(conf < tau))})
        signals[signal] = {"budget_curve": budget_curve(p1_test, p2_test, truth, conf, n_classes, budgets),
                           "over_escalation_ratio": over_escalation_ratio(
                               p1_test, p2_test, truth, conf, n_classes, target_macro_f1),
                           "spearman_with_msp": correlation, "tau_transfer": transfer}
    return {"temperature": temperature, "calibration": calibration, "signals": signals}
