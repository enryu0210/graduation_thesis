"""논문 지표의 정의·방향·재현성을 작은 입력의 알려진 값으로 고정한다."""

import numpy as np
import pytest

import metrics as M


def test_brier_sum_version_and_legacy_keys():
    y = np.array([0, 1])
    perfect = M.compute_metrics(y, y, ["Normal", "Attack"], np.eye(2))
    assert perfect["calibration"]["brier"] == 0.0
    assert isinstance(perfect["calibration"]["brier"], float)
    assert {"ece", "mce"} <= perfect["calibration"].keys()
    assert "pr_auc_macro" in perfect
    assert M.calibration_metrics(y, np.eye(2)[::-1])["brier"] == 2.0
    assert M.calibration_metrics(y, np.full((2, 2), 0.5))["brier"] == 0.5


def test_pr_context_reports_both_positive_classes_and_counts():
    # 동점 점수의 AP 는 양성 비율과 같아 유병률과 양성 방향을 동시에 검증한다.
    y = np.array([1, 0, 0, 2])
    scores = np.full((4, 3), 1 / 3)
    context = M.compute_metrics(y, scores.argmax(1), ["SQLi", "Normal", "CmdI"], scores)["pr_auc_context"]
    assert context == {"attack_prevalence": 0.75, "pr_auc_attack_positive": 0.75,
                       "pr_auc_normal_positive": 0.25, "n_normal": 1, "n_attack": 3}


def test_pr_context_without_normal_or_scores():
    y = np.array([0, 1])
    assert M.compute_metrics(y, y, ["SQLi", "CmdI"], np.eye(2))["pr_auc_context"] is None
    assert "pr_auc_context" not in M.compute_metrics(y, y, ["Normal", "CmdI"])


@pytest.mark.parametrize("label", [0, 1])
def test_pr_context_missing_observed_class(label):
    y = np.full(4, label)
    context = M.compute_metrics(y, y, ["Normal", "Attack"], np.full((4, 2), 0.5))["pr_auc_context"]
    assert context["attack_prevalence"] == label
    assert context["pr_auc_attack_positive"] is None
    assert context["pr_auc_normal_positive"] is None


def test_fe_direction_weight_and_clipping():
    assert M.fe_score(0.9, 10, 5, alpha=1) == 0.9
    assert M.fe_score(0.9, 5, 5) == pytest.approx(0.902)
    assert M.fe_score(0.9, 50, 5) == pytest.approx(0.884)
    assert M.fe_score(0.9, 5, 5) > M.fe_score(0.9, 50, 5)
    assert M.fe_score(0.9, 1, 5, alpha=0) == 1.0


@pytest.mark.parametrize("latency,reference", [(0, 5), (-1, 5), (5, 0), (5, -1), (np.nan, 5), (5, np.inf)])
def test_fe_invalid_latency(latency, reference):
    assert M.fe_score(0.9, latency, reference) is None


@pytest.mark.parametrize("f1,alpha", [(-0.1, 0.98), (1.1, 0.98), (0.9, -1), (0.9, 1.1)])
def test_fe_invalid_weights(f1, alpha):
    with pytest.raises(ValueError):
        M.fe_score(f1, 5, 5, alpha)


def test_bootstrap_reproducibility_and_percentiles():
    y = np.arange(10)
    pred = np.array([0, 1, 2, 3, 4, 5, 0, 0, 0, 0])
    metric = lambda truth, prediction: np.mean(truth == prediction)
    first = M.bootstrap_ci(y, pred, metric, n_resamples=200)
    assert first == M.bootstrap_ci(y, pred, metric, n_resamples=200)
    rng = np.random.default_rng(42)
    values = [metric(y[index], pred[index]) for index in rng.integers(0, 10, (200, 10))]
    assert [first["lo"], first["hi"]] == pytest.approx(np.quantile(values, [0.025, 0.975]))
    assert first["lo"] <= first["point"] <= first["hi"]
    assert first["n_valid"] == first["n_resamples"] == 200


def test_bootstrap_scores_stay_paired_and_invalid_resamples_are_discarded():
    y = np.array([0, 1])

    def metric(truth, prediction, scores):
        # 정답·예측·점수의 같은 행을 재표집해야 점수 기반 CI 가 의미를 갖는다.
        np.testing.assert_array_equal(truth, prediction)
        np.testing.assert_array_equal(truth, scores[:, 0])
        if np.all(truth == 0):
            raise ValueError("양성 표본 없음")
        if np.all(truth == 1):
            return np.nan
        return 0.5

    result = M.bootstrap_ci(y, y, metric, n_resamples=100, y_score=y[:, None])
    rng = np.random.default_rng(42)
    expected_valid = sum(len(set(index)) == 2 for index in rng.integers(0, 2, (100, 2)))
    assert result["n_valid"] == expected_valid
    assert 0 < result["n_valid"] < 100
    assert result["lo"] == result["point"] == result["hi"] == 0.5


def test_bootstrap_no_valid_results():
    result = M.bootstrap_ci([0], [0], lambda *_: np.inf, n_resamples=5)
    assert result["n_valid"] == 0
    assert result["point"] is result["lo"] is result["hi"] is None


@pytest.mark.parametrize("kwargs", [{"n_resamples": 0}, {"n_resamples": 1.5},
                                    {"confidence": 1}, {"seed": None}, {"y_score": []}])
def test_bootstrap_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        M.bootstrap_ci([0], [0], lambda *_: 0.0, **kwargs)


@pytest.mark.parametrize("truth,prediction", [([], []), ([0], [0, 1])])
def test_bootstrap_invalid_samples(truth, prediction):
    with pytest.raises(ValueError):
        M.bootstrap_ci(truth, prediction, lambda *_: 0.0)


def test_wilson_boundaries_and_known_interval():
    zero = M.wilson_ci(0, 10)
    assert zero["lo"] == 0.0
    assert zero["hi"] == pytest.approx(0.2775327998628892)
    full = M.wilson_ci(10, 10)
    assert full["hi"] == 1.0
    assert full["lo"] == pytest.approx(1 - zero["hi"])
    middle = M.wilson_ci(5, 10)
    assert middle["point"] == 0.5
    assert middle["lo"] == pytest.approx(0.236593090512564)
    assert middle["hi"] == pytest.approx(0.763406909487436)
    assert M.wilson_ci(5, 10, confidence=0.99)["lo"] < middle["lo"]


def test_wilson_empty():
    assert M.wilson_ci(0, 0) == {"point": None, "lo": None, "hi": None, "n": 0, "confidence": 0.95}


@pytest.mark.parametrize("successes,n,confidence", [(-1, 10, 0.95), (11, 10, 0.95),
                                                  (0, -1, 0.95), (0.5, 10, 0.95), (1, 10, 1)])
def test_wilson_invalid_input(successes, n, confidence):
    with pytest.raises(ValueError):
        M.wilson_ci(successes, n, confidence)
