"""G6 거리 점수의 방향(클수록 넘김)과 참조 뱅크 상한을 고정한다."""
import numpy as np
import pytest

from src.models import feature_signals as fs


def _data(n=300, seed=0):
    """클래스 3개가 서로 떨어진 가우시안 덩어리."""
    rng = np.random.default_rng(seed)
    centers = np.array([[6, 0, 0, 0], [0, 6, 0, 0], [0, 0, 6, 0]], dtype=float)
    y = np.repeat(np.arange(3), n)
    emb = centers[y] + rng.normal(scale=1.0, size=(len(y), 4))
    return emb, y, centers


def _onehot(pred, k=3):
    p = np.full((len(pred), k), .1 / (k - 1))
    p[np.arange(len(pred)), pred] = .9
    return p


def test_far_points_score_higher():
    emb, y, centers = _data()
    ref = fs.fit_reference(emb, y, 3, bank_per_class=100)
    near, far = centers[0], centers[0] + np.array([0, 0, 0, 15.0])
    out = fs.scores(ref, np.stack([near, far]), _onehot(np.array([0, 0])))
    assert out["maha"][1] > out["maha"][0]
    assert out["knn"][1] > out["knn"][0]


def test_trust_flags_wrong_prediction():
    emb, y, centers = _data()
    ref = fs.fit_reference(emb, y, 3, bank_per_class=100)
    z = np.stack([centers[1], centers[1]])
    # 같은 점을 맞게(1) 예측했을 때보다 틀리게(0) 예측했을 때 넘김 점수가 커야 한다.
    out = fs.scores(ref, z, _onehot(np.array([1, 0])))
    assert out["trust"][1] > out["trust"][0]


def test_shapes_finite_and_bank_cap():
    emb, y, _ = _data()
    ref = fs.fit_reference(emb, y, 3, bank_per_class=100)
    assert np.bincount(ref["bank_y"]).tolist() == [100, 100, 100]
    out = fs.scores(ref, emb[:37], _onehot(y[:37]))
    assert set(out) == set(fs.SCORE_NAMES)
    assert all(v.shape == (37,) and np.isfinite(v).all() for v in out.values())
    latency = fs.measure_latency(ref, emb, _onehot(y), requests=20)
    assert set(latency) == set(fs.SCORE_NAMES) and all(v > 0 for v in latency.values())


def test_missing_class_rejected():
    emb, y, _ = _data()
    with pytest.raises(ValueError):
        fs.fit_reference(emb[y != 2], y[y != 2], 3)


def test_maha_matches_textbook_formula():
    """촐레스키 지름길이 (z−μ)ᵀP(z−μ) 정의와 같은 값을 내는지."""
    from sklearn.covariance import LedoitWolf
    emb, y, _ = _data()
    ref = fs.fit_reference(emb, y, 3, bank_per_class=100)
    means = np.stack([emb[y == c].mean(0) for c in range(3)])
    precision = LedoitWolf().fit(emb - means[y]).precision_
    z = emb[:5]
    direct = np.array([[(v - m) @ precision @ (v - m) for m in means] for v in z]).min(1)
    assert np.allclose(fs.scores(ref, z, _onehot(y[:5]))["maha"], direct)
