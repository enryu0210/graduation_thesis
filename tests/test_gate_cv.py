"""합성 확률 파일로 정렬·목표 일괄 하향·결측 검정 처리를 검증한다."""

import json
import sys

import numpy as np
import pytest

from src.eval import gate_cv


def synthetic_files(tmp_path):
    y = np.array([0, 1, 0, 1, 0, 1])
    arrays = {"classes": np.array(["Normal", "Attack"]), "y": y,
              "label_fingerprint": np.asarray("synthetic")}
    for k, (val, test) in enumerate((([0, 1], [2, 3]), ([2, 3], [4, 5])), start=1):
        arrays[f"val_idx_{k}"] = np.asarray(val)
        arrays[f"test_idx_{k}"] = np.asarray(test)
        for split in ("val", "test"):
            arrays[f"{split}_probs_{k}"] = np.array([[.8, .2], [.2, .8]], dtype=np.float32)
    paths = [tmp_path / "cvprobs_first.npz", tmp_path / "cvprobs_second.npz"]
    for path in paths:
        np.savez(path, **arrays)
    return paths, arrays


@pytest.mark.parametrize("key", ["test_idx_1", "val_idx_2", "classes", "y", "label_fingerprint"])
def test_alignment_mismatch_is_error(tmp_path, key):
    paths, arrays = synthetic_files(tmp_path)
    if key == "label_fingerprint":
        arrays[key] = np.asarray("different")
    else:
        arrays[key] = arrays[key][::-1]
    np.savez(paths[1], **arrays)
    with pytest.raises(ValueError, match=key):
        gate_cv.validate_alignment(*paths)


def test_fold_count_mismatch_is_error(tmp_path):
    paths, arrays = synthetic_files(tmp_path)
    np.savez(paths[1], **{key: value for key, value in arrays.items() if not key.endswith("_2")})
    with pytest.raises(ValueError, match="fold 수"):
        gate_cv.validate_alignment(*paths)


def test_fallback_recomputes_all_folds_and_signals(tmp_path, monkeypatch):
    paths, _ = synthetic_files(tmp_path)
    first, second, folds = gate_cv.validate_alignment(*paths)
    calls = []

    def report(*args):
        target = args[-1]
        calls.append(target)
        # 두 번째 fold만 최초 목표에 실패해도 첫 fold도 다시 계산해야 한다.
        reached = len(calls) != 2
        return {"temperature": target, "signals": {
            signal: {"over_escalation_ratio": {"reached": reached, "ratio": target}}
            for signal in ("msp", *gate_cv.SIGNALS)}}

    monkeypatch.setattr(gate_cv.gate_signals, "signal_report", report)
    result = gate_cv.reports_with_fallback(first, second, folds)
    assert calls == [1., 1., .995, .995]
    assert result["used_offset"] == .005
    assert result["initial_msp_unreached_folds"] == [2]
    for row in result["folds"]:
        for signal in ("msp", *gate_cv.SIGNALS):
            assert row["signals"][signal]["over_escalation_ratio"]["ratio"] == .995


def ratio_reports():
    return [{"fold": k, "signals": {
        name: {"over_escalation_ratio": {"ratio": value}}
        for name, value in zip(("msp", *gate_cv.SIGNALS), values)}}
        for k, values in enumerate(((2, 1, 1.7, 1.2), (3, 1.6, 2.2, 2.5),
                                     (4, 2, 3.9, 2.1), (5, 2.8, 4.3, 4.6),
                                     (6, 2.9, 5.8, 3.)), start=1)]


def test_none_ratio_skips_pair_without_imputation():
    reports = ratio_reports()
    reports[1]["signals"]["margin"]["over_escalation_ratio"]["ratio"] = None
    result = gate_cv.compare_ratios(reports)
    assert result["margin"]["raw_p"] is None
    assert result["margin"]["adjusted_p"] is None
    assert result["margin"]["h_e5_1"] is None
    assert result["margin"]["ratios"][1]["signal"] is None
    assert "결측" in result["margin"]["reason"]
    # 결측 쌍이 있어도 사전 고정한 세 쌍을 보정 모집단으로 유지한다.
    names = ["msp_T", "entropy"]
    expected = gate_cv.holm_bonferroni([result[name]["raw_p"] for name in names] + [1.])
    assert [result[name]["adjusted_p"] for name in names] == expected[:2]


def test_holm_independent_of_signal_and_fold_order():
    reports = ratio_reports()
    first = gate_cv.compare_ratios(reports)
    second = gate_cv.compare_ratios(reports[::-1], signals=gate_cv.SIGNALS[::-1])
    for signal in gate_cv.SIGNALS:
        assert first[signal]["adjusted_p"] == pytest.approx(second[signal]["adjusted_p"])
        assert first[signal]["reduction"] == pytest.approx(second[signal]["reduction"])
        assert first[signal]["h_e5_1"] == second[signal]["h_e5_1"]


def test_synthetic_cli_saves_calibration_and_null_tests(tmp_path, monkeypatch):
    paths, _ = synthetic_files(tmp_path)
    output = tmp_path / "gate_result.json"
    monkeypatch.setattr(sys, "argv", ["gate_cv", "--stage1", str(paths[0]),
                                      "--stage2", str(paths[1]), "--out", str(output)])
    assert gate_cv.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["used_offset"] == 0
    assert not result["fallback_applied"]
    for row in result["folds"]:
        assert row["temperature"] > 0
        assert row["calibration"]["before"]["ece"] >= 0
        assert row["calibration"]["after"]["ece"] >= 0
    assert result["comparisons"]["msp_T"]["raw_p"] is None
