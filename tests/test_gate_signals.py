"""합성 확률로 교정·예산·기존 캐스케이드와의 동등성을 고정한다."""

import json

import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import spearmanr

from src.models.cascade import cascade_apply
from src.models.gate_signals import (
    apply_temperature, fit_temperature, gate_confidence, tau_for_budget,
    cascade_apply_signal, budget_curve, over_escalation_ratio, signal_report,
)


def test_temperature_recovery():
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(30000, 4)) * 2
    true_probs = softmax(logits / 2, axis=1)
    labels = (rng.random(30000)[:, None] > np.cumsum(true_probs, axis=1)).sum(axis=1)
    assert 1.8 <= fit_temperature(softmax(logits, axis=1), labels) <= 2.2


def test_binary_rank_and_manual_signals():
    p = np.column_stack((np.linspace(0.51, 0.99, 100), np.linspace(0.49, 0.01, 100)))
    assert spearmanr(gate_confidence(p, "msp"), gate_confidence(p, "msp_T", 2)).statistic == pytest.approx(1)
    sample = np.array([[0.6, 0.3, 0.1]])
    assert gate_confidence(sample, "margin", 1)[0] == pytest.approx(0.3)
    assert gate_confidence(sample, "entropy", 1)[0] == pytest.approx(sum(x * np.log(x) for x in sample[0]))
    assert np.isfinite(gate_confidence([[1, 0]], "entropy", 0.001)).all()
    np.testing.assert_allclose(apply_temperature(sample, 2), np.sqrt(sample) / np.sqrt(sample).sum())


@pytest.mark.parametrize("signal", ["msp_T", "margin", "entropy", "unknown"])
def test_missing_temperature_or_unknown_signal(signal):
    with pytest.raises(ValueError):
        gate_confidence([[0.6, 0.4]], signal)


@pytest.mark.parametrize("tau", [0, 0.6, 0.9, 1.1])
def test_cascade_exact_msp_equivalence(tau):
    p1 = np.array([[0.6, 0.4], [0.9, 0.1], [0.4, 0.6]], dtype=np.float32)
    p2 = p1[:, ::-1]
    for actual, expected in zip(cascade_apply_signal(p1, p2, gate_confidence(p1, "msp"), tau), cascade_apply(p1, p2, tau)):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("budget", [0, 0.1, 0.25, 0.5, 0.9, 1])
def test_tau_maximal_feasible_rate(budget):
    conf = np.array([-1, -1, 0, 1])
    rate = np.mean(conf < tau_for_budget(conf, budget))
    candidates = [np.mean(conf < tau) for tau in [-1, 0, 1, 2]]
    assert rate == max(value for value in candidates if value <= budget)


def test_perfect_signal_ratio_unreachable_and_zero_oracle():
    y = np.array([0, 1, 0, 1])
    p2 = np.eye(2)[y]
    p1 = np.eye(2)[[1, 0, 0, 1]]
    conf = np.array([0, 0, 1, 1])
    report = over_escalation_ratio(p1, p2, y, conf, 2, 1, step=0.25)
    assert report["ratio"] == 1.0 and report["needed_escalation"] == 0.5
    assert not over_escalation_ratio(p1, p2, y, conf, 2, 1.1, step=0.25)["reached"]
    assert over_escalation_ratio(p2, p2, y, conf, 2, 1, step=0.25)["ratio"] is None


def test_stable_ties_and_absent_class_macro_definition():
    p1 = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    p2 = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
    curve = budget_curve(p1, p2, np.array([1, 0]), [1, 1], 3, [0.5])
    assert curve[0]["k"] == 1
    assert curve[0]["macro_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("probs,y", [([], []), ([[0.5, 0.5]], []), ([[0.1, 0.2]], [0]),
                                        ([[np.nan, 0.5]], [0]), ([[0.5, 0.5]], [2])])
def test_fit_invalid_inputs(probs, y):
    with pytest.raises(ValueError):
        fit_temperature(probs, y)


@pytest.mark.parametrize("budget", [-0.1, 1.1, np.nan])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        tau_for_budget([0.5], budget)


def test_float32_budget_endpoints():
    conf = np.array([0.6, 0.9], dtype=np.float32)
    assert np.mean(conf < tau_for_budget(conf, 1)) == 1
    assert np.mean(conf < tau_for_budget(conf, 0)) == 0


def test_constant_report_is_strict_json():
    probs = np.full((2, 2), 0.5)
    report = signal_report(probs, np.array([0, 1]), probs, probs, np.array([0, 1]), 2, 1)
    json.dumps(report, allow_nan=False)
    assert all(value["spearman_with_msp"] is None for value in report["signals"].values())


def test_report_json_and_val_tau_transfer():
    val = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    test = np.array([[0.51, 0.49], [0.49, 0.51], [0.6, 0.4], [0.4, 0.6]])
    y = np.array([0, 1, 0, 1])
    report = signal_report(val, y, test, test, y, 2, 1, budgets=(0.25,))
    json.dumps(report, allow_nan=False)
    for name, values in report["signals"].items():
        transfer = values["tau_transfer"][0]
        expected_tau = tau_for_budget(gate_confidence(val, name, report["temperature"]), 0.25)
        assert transfer["tau"] == expected_tau
        assert transfer["escalation_rate"] == np.mean(gate_confidence(test, name, report["temperature"]) < expected_tau)
    assert set(report["calibration"]["before"]) == {"ece", "mce", "brier"}
