"""학습 없이 운영점 선택·비용 회계·출처 행 정렬을 고정한다."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.eval import cascade_tradeoff as tradeoff


def test_operating_point_selects_smallest_budget_using_val_only():
    rows = [{"budget": .2, "val_macro_f1": .95, "test_macro_f1": 1.},
            {"budget": .01, "val_macro_f1": .945, "test_macro_f1": .1},
            {"budget": 0, "val_macro_f1": .94, "test_macro_f1": 1.}]
    assert tradeoff.select_operating_point(rows, .95) is rows[1]


def test_operating_point_missing_is_none():
    assert tradeoff.select_operating_point([{"budget": 1, "val_macro_f1": .9}], .95) is None
    assert tradeoff.select_operating_point([], .95) is None


@pytest.mark.parametrize("prevalence,expected", [(0, .1), (1, .8), (.01, .107)])
def test_reweighted_formula(prevalence, expected):
    assert tradeoff.reweighted_escalation(.8, .1, prevalence) == pytest.approx(expected)


def test_latency_formula():
    result = tradeoff.latency_point(2, 10, .2)
    assert result["cascade_ms"] == 4
    assert result["speedup"] == 2.5


@pytest.mark.parametrize("args", [(-1, 10, .2), (1, 0, .2), (1, 10, 1.1)])
def test_invalid_latency_rejected(args):
    with pytest.raises(ValueError):
        tradeoff.latency_point(*args)


def strata_fixture():
    test = pd.DataFrame({"row_id": [20, 10, 30], "label": ["Normal", "Attack", "Normal"]})
    strata = pd.DataFrame({"split": ["test", "val", "test"], "row_id": [30, 999, 20],
                           "normal_source": ["scanner", "mixed", "real_user"]})
    return test, strata, np.array([0, 1, 0]), ["Normal", "Attack"]


def test_strata_aligns_by_row_id_and_filters_split():
    aligned = tradeoff.align_normal_strata(*strata_fixture())
    assert aligned[0] == "real_user"
    assert pd.isna(aligned[1])
    assert aligned[2] == "scanner"
    fpr = tradeoff.srbh_normal_strata.fpr_by_stratum(
        ["Normal", "Attack", "Normal"], ["Attack", "Attack", "Normal"], aligned)
    assert fpr["real_user"]["false_alarms"] == 1
    assert fpr["scanner"]["false_alarms"] == 0


@pytest.mark.parametrize("problem", ["missing", "wrong_id", "duplicate", "unknown", "length", "label"])
def test_strata_mismatch_is_error(problem):
    test, strata, y, classes = strata_fixture()
    if problem == "missing":
        strata = strata.iloc[:2]
    elif problem == "wrong_id":
        strata.loc[0, "row_id"] = 100
    elif problem == "duplicate":
        strata.loc[0, "row_id"] = 20
    elif problem == "unknown":
        strata.loc[0, "normal_source"] = None
    elif problem == "length":
        y = y[:2]
    else:
        test.loc[0, "label"] = "Attack"
    with pytest.raises(ValueError):
        tradeoff.align_normal_strata(test, strata, y, classes)


def test_budget_tau_uses_val_and_preserves_saturated_ties():
    p1_val = np.array([[1, 0], [1, 0], [.6, .4], [.4, .6]], dtype=np.float32)
    p2 = np.array([[.1, .9], [.9, .1], [.1, .9], [.9, .1]], dtype=np.float32)
    p1_test = np.array([[.51, .49], [.51, .49], [1, 0], [0, 1]], dtype=np.float32)
    y = np.array([1, 0, 1, 0])
    rows = tradeoff.budget_table(p1_val, p2, y, p1_test, p2, y, ["Normal", "Attack"], None)
    for row in rows:
        assert row["tau"] == tradeoff.gate_signals.tau_for_budget(p1_val.max(1), row["budget"])
    assert rows[0]["val_escalation_rate"] == 0
    assert rows[0]["test_escalation_rate"] == .5
    assert rows[-1]["test_escalation_rate"] == 1


def test_missing_strata_stops_with_generation_command(tmp_path, monkeypatch):
    monkeypatch.setattr(tradeoff, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="srbh_normal_strata.py"):
        tradeoff.load_strata("srbh_4class", None, np.array([0]), ["Normal", "Attack"])
    assert tradeoff.load_strata("srbh_4class", 10, np.array([0]), ["Normal", "Attack"])[0] is None


def test_output_tag_covers_checkpoint_and_measurement_axes():
    args = SimpleNamespace(track="srbh_4class", text="raw", balance=True, channels="rgb",
                           max_len=2304, limit=None, seed=42, devices=["cpu"], batches=[1, 32])
    encoders = ("raw_byte", "char_class", "local_entropy")
    baseline = tradeoff.result_tag(args, encoders)
    assert baseline.startswith("srbh_4class_cnn_raw_rgb_bal")
    for key, value in (("max_len", 100), ("limit", 2000), ("seed", 1),
                       ("devices", ["cuda", "cpu"]), ("batches", [1])):
        changed = SimpleNamespace(**vars(args))
        setattr(changed, key, value)
        assert tradeoff.result_tag(changed, encoders) != baseline
    assert tradeoff.result_tag(args, ("raw_byte", "raw_byte", "raw_byte")) != baseline
