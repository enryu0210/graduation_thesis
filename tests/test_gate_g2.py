"""합성 확률로 클래스별 탐욕 경로·지표·CLI 계약을 검증한다."""

import json
import sys

import numpy as np
import pytest

from src.eval import gate_g2
from src.eval.metrics import compute_metrics
from src.models import gate_signals as gates


def synthetic_probabilities():
    # 0 예측은 모두 틀리고 1 예측은 모두 맞으므로 첫 예산의 개선 방향이 명확하다.
    first = np.array([[.6, .4], [.7, .3], [.4, .6], [.3, .7]], dtype=np.float32)
    second = np.array([[.1, .9]] * 4, dtype=np.float32)
    return first, second, np.ones(4, dtype=int)


def test_equal_thresholds_match_single_gate():
    first, second, _ = synthetic_probabilities()
    conf = first.max(1)
    for tau in (0., .65, 1., np.nextafter(1., np.inf)):
        expected = gates.cascade_apply_signal(first, second, conf, tau)[2]
        np.testing.assert_array_equal(gates.apply_classwise(first, conf, [tau, tau]), expected)


def test_greedy_first_class_and_monotone_full_transfer():
    first, second, truth = synthetic_probabilities()
    path = gates.classwise_tau_path(first, second, truth, 2, step=.25)
    assert path[0]["allocations"] == [0, 0]
    assert path[1]["allocations"] == [1, 0]
    assert np.all(np.diff([p["allocations"] for p in path], axis=0) >= 0)
    assert path[-1]["val_escalation_rate"] == 1.
    mask = gates.apply_classwise(first, first.max(1), path[-1]["taus"])
    np.testing.assert_array_equal(np.where(mask, second.argmax(1), first.argmax(1)), second.argmax(1))
    for point in path:
        mask = gates.apply_classwise(first, first.max(1), point["taus"])
        pred = np.where(mask, second.argmax(1), first.argmax(1))
        assert point["val_macro_f1"] == pytest.approx(
            compute_metrics(truth, pred, ["Normal", "Attack"])["macro_f1"], abs=1e-12, rel=0)


@pytest.mark.parametrize("n_classes", [2, 4, 7])
def test_fast_f1_matches_common_metrics(n_classes):
    rng = np.random.default_rng(42)
    for size in (1, 10, 1000):
        truth, pred = rng.integers(0, n_classes, size=(2, size))
        expected = compute_metrics(truth, pred, [str(i) for i in range(n_classes)])["macro_f1"]
        assert gates._macro_f1_fast(truth, pred, n_classes) == pytest.approx(expected, abs=1e-12, rel=0)


@pytest.mark.parametrize("taus", [[.5], [.5, np.inf], [.5, np.nan], ["a", "b"], [[.5, .5]]])
def test_invalid_thresholds_rejected(taus):
    first, _, _ = synthetic_probabilities()
    with pytest.raises(ValueError):
        gates.apply_classwise(first, first.max(1), taus)


def test_absent_predicted_class_rejected():
    first, second, truth = synthetic_probabilities()
    with pytest.raises(ValueError, match="예측 클래스"):
        gates.classwise_tau_path(second, first, truth, 2)


@pytest.mark.parametrize("rate", [0., 1.])
def test_deployment_speedup_endpoints(rate):
    result = gate_g2.deployment_cost(rate, rate, .033, .350, 1.114)
    assert result["speedup"] == pytest.approx(1.114 / (.350 + rate * 1.114))


def synthetic_files(tmp_path):
    truth = np.tile([0, 1, 0, 1], 3)
    first_probs = np.array([[.6, .4], [.4, .6], [.7, .3], [.3, .7]], dtype=np.float32)
    second_probs = np.array([[.9, .1], [.1, .9]] * 2, dtype=np.float32)
    arrays = {"classes": np.array(["Normal", "Attack"]), "y": truth,
              "label_fingerprint": np.asarray("synthetic")}
    for k in range(1, 3):
        arrays[f"val_idx_{k}"] = np.arange((k - 1) * 4, k * 4)
        arrays[f"test_idx_{k}"] = np.arange(k * 4, (k + 1) * 4)
        for split in ("val", "test"):
            arrays[f"{split}_probs_{k}"] = first_probs
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, **arrays)
    for key in arrays:
        if "probs" in key:
            arrays[key] = second_probs
    np.savez(second, **arrays)
    return first, second


def test_cli_json_summary_and_no_overwrite(tmp_path, monkeypatch, capsys):
    first, second = synthetic_files(tmp_path)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["gate_g2", "--stage1", str(first), "--stage2", str(second),
                                      "--out", str(output)])
    assert gate_g2.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == result["summary"]
    assert {"folds", "reduction", "p", "h_g2", "operating_points"} <= result["summary"].keys()
    assert "taus_by_class" in result["summary"]["operating_points"]["g2"]
    before = output.read_bytes()
    assert gate_g2.main() == 1
    assert output.read_bytes() == before


def test_fallback_lowers_all_folds_and_gates(tmp_path, monkeypatch):
    first_path, second_path = synthetic_files(tmp_path)
    first, second, folds = gate_g2.validate_alignment(first_path, second_path)
    rows = gate_g2.fold_curves(first, second, folds, .5)
    # 첫 fold만 목표에 미달해도 두 fold·두 게이트에 같은 하향 목표가 필요하다.
    for point in rows[0]["curves"]["b0"]:
        point["test_macro_f1"] = .997
    monkeypatch.setattr(gate_g2, "fold_curves", lambda *args: rows)
    result = gate_g2.reports_with_fallback(first, second, folds, step=.5)
    assert result["initial_b0_unreached_folds"] == [1]
    assert result["fallback_applied"]
    assert result["used_offset"] == .005
    assert all(row["target_macro_f1"] == .995 for row in result["folds"])
    assert all(row["gates"][name]["reached"] for row in result["folds"] for name in ("b0", "g2"))


def test_missing_ratio_skips_test(tmp_path):
    paths = synthetic_files(tmp_path)
    first, second, folds = gate_g2.validate_alignment(*paths)
    reports = gate_g2.reports_with_fallback(first, second, folds)["folds"]
    reports[0]["gates"]["g2"]["ratio"] = None
    summary = gate_g2.summarize(reports, first["classes"].tolist())
    assert summary["p"] is None and summary["h_g2"] is None
    assert "결측" in summary["reason"]
