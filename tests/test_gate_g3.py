"""합성 데이터로 고정 split 라우터의 누출 방지와 출력 계약을 검증한다."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.eval import gate_g3 as g3
from src.models import gate_signals as gates


def synthetic():
    # 같은 1차 확률에서도 원문의 표식이 복구 가능 행을 완벽히 구별한다.
    y = np.tile(np.arange(4), 40)
    recover = np.tile(np.repeat([False, True], 4), 20)
    pred = np.where(recover, (y + 1) % 4, y)
    p1 = np.full((len(y), 4), .1)
    p1[np.arange(len(y)), pred] = .7
    p2 = np.full_like(p1, .02)
    p2[np.arange(len(y)), y] = .94
    texts = ["%" * 50 if value else "abc" for value in recover]
    return y, p1, p2, texts, recover


@pytest.mark.parametrize("text, column, expected", [("", 0, 0.), ("a" * 2305, 4, 1.), ("%41", 2, 1 / 3), ("한" * 769, 4, 1.)])
def test_text_features(text, column, expected):
    features = g3.text_features([text])
    assert features.shape == (1, 5)
    assert np.isfinite(features).all()
    assert features[0, column] == pytest.approx(expected)


def test_special_characters_and_digit_denominator():
    result = g3.text_features(["%41", "%'\"<>();=&|$`{}", ""])
    assert result[0, 3] == pytest.approx(2 / 3)
    assert result[1, 1] == 1.
    np.testing.assert_array_equal(result[2], np.zeros(5))


def fitted():
    y, p1, p2, texts, target = synthetic()
    features = g3.router_features(p1, g3.text_features(texts))
    return (y, p1, p2, texts, features, g3.cross_fit_router(features, y, target, features))


def test_router_reduces_operating_transfer():
    y, p1, p2, texts, features, (sv, st, models, splits) = fitted()
    curve = g3.router_curve(p1, p2, y, p1, p2, y, sv, st, .05)
    chosen = min((row for row in curve if row["val_macro_f1"] >= .995), key=lambda row: row["val_escalation_rate"])
    b0_rates = []
    for budget in np.r_[np.arange(0, 1, .05), 1.]:
        tau = gates.tau_for_budget(p1.max(1), budget)
        pred, _, mask = gates.cascade_apply_signal(p1, p2, p1.max(1), tau)
        if gates._macro_f1_fast(y, pred, 4) >= .995:
            b0_rates.append(mask.mean())
    assert chosen["val_escalation_rate"] < min(b0_rates)


def test_cross_fit_rows_are_unseen():
    y, p1, p2, texts, features, (sv, st, models, splits) = fitted()
    seen = np.zeros(len(y), dtype=int)
    for model, (train, held) in zip(models, splits):
        assert not np.intersect1d(train, held).size
        assert len(train) + len(held) == len(y)
        seen[held] += 1
        # 실제 학습 행의 중심값과 별도로 재적합한 점수를 함께 대조한다.
        np.testing.assert_allclose(model.steps[0][1].mean_, features[train].mean(0))
        clone = g3.make_pipeline(g3.StandardScaler(), g3.LogisticRegression(C=1., max_iter=2000))
        target = (p1.argmax(1) != y) & (p2.argmax(1) == y)
        clone.fit(features[train], target[train])
        np.testing.assert_allclose(sv[held], clone.predict_proba(features[held])[:, 1])
    np.testing.assert_array_equal(seen, np.ones(len(y)))
    np.testing.assert_allclose(st, np.mean([m.predict_proba(features)[:, 1] for m in models], axis=0))


def test_folded_weights_and_latency():
    y, p1, p2, texts, features, (_, _, models, _) = fitted()
    for model in models:
        weight, bias = g3.folded_weights(model)
        np.testing.assert_allclose(expit(features @ weight + bias), model.predict_proba(features)[:, 1], atol=1e-9, rtol=0)
    assert g3.measure_router_latency(texts, p1, models, 5, 42) > 0


def test_identical_bootstrap_zero():
    y, p1, p2, texts, target = synthetic()
    result = g3.bootstrap_comparison(y, target, y, target, y, 4, 25)
    assert result["escalation_difference_ci"] == [0., 0.]
    assert result["macro_f1_difference_ci"] == [0., 0.]


def test_label_alignment_rejected(monkeypatch):
    monkeypatch.setattr(g3.data_text, "load_text_split", lambda *a: (["a", "b"], pd.Series(["Normal", "Attack"])))
    with pytest.raises(ValueError, match="정렬"):
        g3.aligned_texts("synthetic", "val", np.array([1, 0]), ["Normal", "Attack"])


@pytest.mark.parametrize("target", [False, True])
def test_constant_target_rejected(target):
    y, p1, p2, texts, recover = synthetic()
    features = g3.router_features(p1, g3.text_features(texts))
    with pytest.raises(ValueError, match="표적"):
        g3.cross_fit_router(features, y, np.full(len(y), target), features)


@pytest.mark.parametrize("emb", [np.zeros((2, 3)), np.full((160, 2), np.nan), np.zeros(160)])
def test_invalid_embedding_rejected(emb):
    y, p1, p2, texts, recover = synthetic()
    with pytest.raises(ValueError, match="임베딩"):
        g3.router_features(p1, g3.text_features(texts), emb)


@pytest.mark.parametrize("embedding", [False, True])
def test_cli_complete_json(tmp_path, monkeypatch, capsys, embedding):
    y, p1, p2, texts, target = synthetic()
    classes = np.array(["Normal", "SQLInjection", "CodeInjection", "CommandInjection"])
    probs = tmp_path / "probs.npz"
    np.savez(probs, classes=classes, y_val=y, y_test=y, p1_val=p1, p2_val=p2, p1_test=p1, p2_test=p2)
    monkeypatch.setattr(g3.data_text, "load_text_split", lambda *a: (texts, pd.Series(classes[y])))
    monkeypatch.setattr(g3.cascade_tradeoff, "load_strata", lambda *a: (None, "합성 입력에는 층 CSV 없음"))
    output = tmp_path / "result.json"
    argv = ["gate_g3", "--probs", str(probs), "--step", ".1", "--n-boot", "5", "--latency-requests", "5", "--out", str(output)]
    if embedding:
        emb = tmp_path / "emb.npz"
        np.savez(emb, val=target[:, None].astype(float), test=target[:, None].astype(float))
        argv.extend(["--embeddings", str(emb)])
    monkeypatch.setattr(sys, "argv", argv)
    assert g3.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == result["summary"]
    assert {"b0", "g2", "g3_lite"} <= result["curves"].keys()
    assert ("g3_full" in result["summary"]["gates"]) == embedding
    assert result["summary"]["label_alignment"]["passed"]
    for name in ("g2", "g3_lite"):
        assert "bootstrap" in result["summary"]["gates"][name]
        assert "hypothesis" in result["summary"]["gates"][name]
    before = output.read_bytes()
    assert g3.main() == 1
    assert output.read_bytes() == before
