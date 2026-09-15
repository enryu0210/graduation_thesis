"""논문 지표의 분모와 미성공 제외 규칙을 합성 입력으로 고정한다."""
import json
from unittest.mock import patch

import numpy as np
import pytest

import run_evasion as R
import compare_evasion as C


def test_aua_definition_and_empty_mask():
    pred = np.array([0, 1, 0, 2])
    mask = np.ones(4, dtype=bool)
    # 원본 예시는 정답이 두 개다. 오분류 세 개인 별도 입력도 확인한다.
    for y, expected in [(np.array([1, 1, 2, 2]), 0.5),
                        (np.array([1, 1, 2, 1]), 0.25)]:
        assert R.accuracy_under_attack(pred, mask, y) == expected
        assert R.accuracy_under_attack(pred, mask, y) + R.any_misclass(pred, mask, y) == 1
    assert np.isnan(R.accuracy_under_attack(pred, ~mask, y))
    assert "ASR" in R.benign_evasion.__doc__
    assert "ASR 이 아니다" in R.any_misclass.__doc__


def test_single_preserves_keys(monkeypatch):
    monkeypatch.setattr(R.PS, "ALL_TECHNIQUES", ["합성"])
    monkeypatch.setattr(R.PS, "mutate_single", lambda *args: (["a", "b"], np.array([True, False])))
    row, = R.run_single(lambda _: np.array([0, 0]), ["a", "b"],
                        ["SQLInjection", "Normal"], np.array([1, 0]), np.array([1, 0]), 0, 42)
    for prefix, expected in [("asr", (0, 1, 1)), ("anymis", (0, 1, 1)), ("aua", (1, 0, -1))]:
        assert tuple(row[f"{prefix}_{suffix}"] for suffix in ("clean", "mutated", "delta")) == expected


@pytest.mark.parametrize("mode", ["nonmonotonic", "never", "empty", "zero"])
def test_budget_summary(monkeypatch, mode):
    applied = np.array([True, True, True, False]) if mode != "empty" else np.zeros(4, dtype=bool)
    monkeypatch.setattr(R.PS, "mutate_stacked", lambda texts, labels, k, rng: (k, applied))
    # 첫 샘플은 k=1 성공 후 실패, 둘째는 k=3 성공, 셋째는 항상 실패한다.
    predictions = {1: [0, 1, 1, 0], 2: [1, 1, 1, 0], 3: [1, 0, 1, 0]}
    predict = lambda k: np.array(predictions[k] if mode == "nonmonotonic" else [1, 1, 1, 0])
    rows, summary = R.run_stacked(predict, ["a"] * 4, ["SQLInjection"] * 3 + ["Normal"],
                                  np.array([1, 1, 1, 0]), np.array([0, 1, 1, 0]),
                                  0, 0 if mode == "zero" else 3, 42)
    assert all("aua_mutated" in row for row in rows)
    assert set(summary) == {"amb_mean", "amb_median", "n_succeeded", "n_never_succeeded",
                            "n_applied", "success_rate_within_budget", "definition"}
    assert "AVGQ 와 다르다" in summary["definition"]
    if mode == "nonmonotonic":
        assert summary["amb_mean"] == summary["amb_median"] == 2
        assert summary["n_succeeded"] == 2
        assert summary["n_never_succeeded"] == 1
        assert summary["success_rate_within_budget"] == pytest.approx(2 / 3)
    else:
        assert summary["amb_mean"] is summary["amb_median"] is None
        assert summary["n_succeeded"] == 0
        assert summary["n_never_succeeded"] == (3 if mode == "never" else 0)


@pytest.mark.parametrize("has_aua", [False, True])
def test_loader_compatibility(has_aua):
    row = {"budget_k": 1, "asr_mutated": 0.2, "anymis_mutated": 0.3}
    if has_aua:
        row["aua_mutated"] = 0.6
    # 파일 생성 없이 신구 JSON 스키마의 우선순위를 검증한다.
    with patch.object(C.Path, "read_text", return_value=json.dumps({"results": [row]})):
        curve, _ = C._parse_stacked(C.Path("unused.json"))
    assert curve[2] == [0.6 if has_aua else 0.7]


def test_existing_json_and_plot_labels(monkeypatch):
    paths = list(C.RESULTS_DIR.glob("evasion_*_stacked.json"))
    assert paths
    for path in paths:
        curve, doc = C._parse_stacked(path)
        assert curve[2] == [r["aua_mutated"] if "aua_mutated" in r else 1 - r["anymis_mutated"]
                            for r in doc["results"]]
    # 논문 그림을 덮어쓰지 않고 실제 렌더링 경로의 라벨을 검사한다.
    from matplotlib.figure import Figure
    def check_figure(fig, *args, **kwargs):
        fig.canvas.draw()
        for ax in fig.axes:
            labels = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
            labels += [t.get_text() for t in ax.get_legend().get_texts()]
            assert all(label.isascii() for label in labels)
        assert fig._suptitle.get_text().isascii()
    monkeypatch.setattr(Figure, "savefig", check_figure)
    monkeypatch.setattr(C.Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(C.sys, "argv", ["compare_evasion"])
    C.main()
