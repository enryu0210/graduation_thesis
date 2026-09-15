"""출처 정렬과 합산을 합성 데이터로 검증하여 기존 CV 표본을 보존한다."""

import sys

import numpy as np
import pandas as pd
import pytest

from src.eval import cross_validate as cv
import data_text


@pytest.fixture
def strata_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "PROJECT_ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(data_text, "PROCESSED_DIR", processed)
    # split 사이 row_id가 같고 층 CSV 순서가 달라도 올바르게 연결되어야 한다.
    for split in cv.SPLITS:
        pd.DataFrame({"row_id": [2, 1, 3], "label": ["Normal", "Normal", "SQLInjection"]}).to_csv(
            data_text.csv_path("srbh_4class", split), index=False)
    path = processed / "srbh_4class_normal_strata.csv"
    pd.DataFrame({
        "split": ["val", "train", "test", "train", "val", "test"],
        "row_id": [1, 2, 2, 1, 2, 1],
        "normal_source": ["mixed", "real_user", "scanner", "scanner", "real_user", "mixed"],
    }).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("exclude_test", [True, False])
def test_strata_alignment(strata_files, exclude_test):
    expected = ["real_user", "scanner", "", "real_user", "mixed", ""]
    if not exclude_test:
        expected += ["scanner", "mixed", ""]
    assert cv.load_pool_strata("srbh_4class", exclude_test).tolist() == expected


@pytest.mark.parametrize("problem", ["missing", "invalid", "duplicate", "column"])
def test_invalid_strata(strata_files, problem):
    frame = pd.read_csv(strata_files)
    if problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "invalid":
        frame.loc[0, "normal_source"] = "unknown"
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame = frame.drop(columns="normal_source")
    frame.to_csv(strata_files, index=False)
    with pytest.raises(ValueError, match="Normal 출처 층"):
        cv.load_pool_strata("srbh_4class", True)


@pytest.mark.parametrize("model", ["charcnn", "tfidf_logreg", "cnn"])
def test_missing_csv_stops_before_training(strata_files, monkeypatch, model):
    strata_files.unlink()
    monkeypatch.setattr(sys, "argv", ["cv", "--model", model, "--track", "srbh_4class",
                                      "--fields", "F1", "--exclude-test"])
    monkeypatch.setattr(cv, "load_text_pool", lambda *a, **kw: (["a"], np.array([0]), ["Normal"]))
    monkeypatch.setattr(cv, "load_image_pool", lambda *a, **kw: (np.zeros((1, 2, 2)), np.array([0]), ["Normal"]))

    def forbidden(*args, **kwargs):
        pytest.fail("층 CSV 검증 전에 학습 또는 시퀀스 인코딩이 시작되었습니다.")

    monkeypatch.setattr(cv, "run_fold_torch", forbidden)
    monkeypatch.setattr(cv, "run_fold_tfidf", forbidden)
    monkeypatch.setattr(cv, "encode_byte_matrix_int16", forbidden)
    with pytest.raises(FileNotFoundError, match="Normal 출처 층 CSV가 없습니다"):
        cv.main()


@pytest.mark.parametrize("texts,labels,expected", [
    (["a", "a", "b", "b"], [0, 1, 0, 0], (1, 2)),
    (["a", "a", "a"], [0, 1, 0], (1, 3)),
    (["a", "a"], [0, 0], (0, 0)),
    ([], [], (0, 0)),
])
def test_input_conflicts(texts, labels, expected):
    assert cv.count_input_conflicts(texts, labels) == {
        "n_conflicting_texts": expected[0], "n_conflicting_rows": expected[1]}


def test_pooled_counts_and_wilson():
    classes = ["Normal", "SQLInjection"]
    reports = [cv.fold_strata_report(
        np.array([0, 0, 0, 1]), np.array(pred), classes,
        np.array(["real_user", "real_user", "mixed", ""]))
        for pred in ([0, 1, 0, 0], [1, 1, 1, 1])]
    pooled = cv.pool_strata_reports([{"fpr_by_stratum": report} for report in reports])
    for name in pooled:
        n = sum(report[name]["n"] for report in reports)
        alarms = sum(report[name]["false_alarms"] for report in reports)
        assert pooled[name] == {"n": n, "false_alarms": alarms, "fpr": cv.M.wilson_ci(alarms, n)}
    assert pooled["real_user"]["n"] == 4
    assert pooled["real_user"]["false_alarms"] == 3
    assert pooled["scanner"]["fpr"]["point"] is None
