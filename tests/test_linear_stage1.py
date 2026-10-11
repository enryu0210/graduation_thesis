"""E3·T11 분석 보조 함수의 최소 점검 (src/eval/linear_stage1.py)."""
import numpy as np
import pytest

from src.eval import linear_stage1 as LS


def _probs(pred, n_classes=2):
    return np.eye(n_classes)[np.asarray(pred)]


def test_q_statistic_is_one_when_errors_coincide_and_negative_when_complementary():
    y = np.array([0, 0, 1, 1])
    same = {"y": y, "test_idx_1": np.arange(4), "test_probs_1": _probs([0, 1, 1, 0])}
    assert LS.q_statistic(same, same, [1]) == pytest.approx(1.0)
    # 한쪽이 틀린 곳을 다른 쪽이 맞히면(오류가 겹치지 않으면) Q 는 −1 이다.
    other = {"y": y, "test_idx_1": np.arange(4), "test_probs_1": _probs([1, 0, 0, 1])}
    assert LS.q_statistic(same, other, [1]) == pytest.approx(-1.0)


def test_versus_reports_gain_of_other_over_base_and_skips_unreachable_folds():
    base = {"per_fold_speedup": {"b0": [1.0, 1.0]}, "b0": {"macro_f1": 0.980}}
    other = {"per_fold_speedup": {"b0": [1.2, 1.3]}, "b0": {"macro_f1": 0.978}}
    row = LS.versus(base, other)
    assert row["speedup_gain"] == pytest.approx(0.25)
    assert row["f1_diff_pp"] == pytest.approx(-0.2)
    assert LS.versus({"reason": "val 목표 미달"}, other) is None
