"""1차 CNN 내부 표현(GAP 128차원) 공간의 학습 없는 거리 점수 — G6 넘김 신호(docs/14 §8.14).

msp·margin·entropy 는 모두 1차 확률 p1 의 함수라 서로 거의 같은 순위를 냈다(G1).
여기 점수들은 p1 이 버린 정보(표현이 학습 분포의 어디쯤 있는가)를 쓴다.
모든 점수는 **클수록 넘김** 방향이다. 정의는 문헌 그대로 가져온다(사후 설계 방지):
- maha  : Lee et al. NeurIPS'18 — 클래스 평균까지의 최소 마할라노비스 거리(공유 공분산)
- rmaha : Ren et al. 2021 — 위 거리에서 전체 단일 가우시안까지의 거리를 뺀 상대 거리
- knn   : Sun et al. ICML'22 — L2 정규화 표현의 k 번째 최근접 거리
- trust : Jiang et al. NeurIPS'18 — (예측 외 클래스 최근접 / 예측 클래스 최근접)의 음수
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.covariance import LedoitWolf

SCORE_NAMES = ("maha", "rmaha", "knn", "trust")
KNN_K = 50
CHUNK_ROWS = 4096   # 거리 행렬을 나눠 계산해 (질의×뱅크) 전체를 메모리에 올리지 않는다


def _embeddings(emb, name="표현"):
    values = np.asarray(emb, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name}은 비어 있지 않은 유한한 N×D 배열이어야 합니다.")
    return values


def _normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def fit_reference(train_emb, train_y, n_classes, bank_per_class=2000, seed=42):
    """train 표현만으로 참조 통계를 만든다 — val·test 는 τ 선택·평가용이라 참조에 넣지 않는다."""
    emb = _embeddings(train_emb, "train 표현")
    labels = np.asarray(train_y)
    if labels.shape != (len(emb),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("train 라벨은 표현 행 수와 같은 정수 배열이어야 합니다.")
    present = np.unique(labels)
    if not np.array_equal(present, np.arange(n_classes)):
        raise ValueError("train 참조에 모든 클래스가 있어야 합니다.")
    means = np.stack([emb[labels == c].mean(axis=0) for c in range(n_classes)])
    # 공유 공분산은 클래스 중심화 후 추정한다(Lee'18). 128차원·소수 클래스에서도 역행렬이
    # 안정하도록 Ledoit–Wolf 수축을 쓴다.
    precision = LedoitWolf().fit(emb - means[labels]).precision_
    background = LedoitWolf().fit(emb)
    # P = L·Lᵀ 로 분해해 두면 (z−μ)ᵀP(z−μ) = ‖(z−μ)L‖² 이라 요청마다 128×128 곱 한 번이면 된다.
    chol = np.linalg.cholesky(precision)
    bg_chol = np.linalg.cholesky(background.precision_)
    rng = np.random.default_rng(seed)
    bank, bank_y = [], []
    for c in range(n_classes):
        rows = np.flatnonzero(labels == c)
        if len(rows) > bank_per_class:
            rows = np.sort(rng.choice(rows, bank_per_class, replace=False))
        bank.append(_normalize(emb[rows]))
        bank_y.append(np.full(len(rows), c))
    sizes = [len(rows) for rows in bank]
    return {"chol": chol, "means_t": means @ chol,
            "bg_chol": bg_chol, "bg_mean_t": background.location_[None] @ bg_chol,
            "bank": np.concatenate(bank), "bank_y": np.concatenate(bank_y),
            "bank_starts": np.r_[0, np.cumsum(sizes)[:-1]], "n_classes": n_classes}


def _class_mahalanobis(emb, means_t, chol):
    """미리 변환한 평균(μL)과의 유클리드 거리² = 마할라노비스 거리²."""
    zt = emb @ chol
    return ((zt[:, None, :] - means_t[None]) ** 2).sum(axis=-1)


def _class_nearest(dist, starts):
    """뱅크가 클래스 순서로 이어져 있으므로 구간별 최솟값 = 클래스별 최근접 거리."""
    return np.minimum.reduceat(dist, starts, axis=1)


def _bank_distances(normed, bank):
    """정규화 벡터끼리는 ‖a−b‖² = 2 − 2a·b."""
    return np.sqrt(np.maximum(2 - 2 * normed @ bank.T, 0))


def scores(ref, emb, p1):
    """이름 → 점수 배열(클수록 넘김). ŷ 는 1차 예측(p1.argmax)."""
    emb = _embeddings(emb)
    pred = np.asarray(p1).argmax(axis=1)
    if len(pred) != len(emb):
        raise ValueError("표현과 확률의 행 수가 다릅니다.")
    out = {name: np.empty(len(emb)) for name in SCORE_NAMES}
    bank, bank_y = ref["bank"], ref["bank_y"]
    if len(bank) < KNN_K:
        raise ValueError(f"kNN 뱅크가 k={KNN_K} 보다 작습니다.")
    for start in range(0, len(emb), CHUNK_ROWS):
        part = slice(start, start + CHUNK_ROWS)
        z = emb[part]
        d_class = _class_mahalanobis(z, ref["means_t"], ref["chol"])
        d_bg = _class_mahalanobis(z, ref["bg_mean_t"], ref["bg_chol"])[:, 0]
        out["maha"][part] = d_class.min(axis=1)
        out["rmaha"][part] = (d_class - d_bg[:, None]).min(axis=1)
        dist = _bank_distances(_normalize(z), bank)
        out["knn"][part] = np.partition(dist, KNN_K - 1, axis=1)[:, KNN_K - 1]
        nearest = _class_nearest(dist, ref["bank_starts"])
        own = nearest[np.arange(len(z)), pred[part]]
        other = nearest.copy()
        other[np.arange(len(z)), pred[part]] = np.inf
        out["trust"][part] = -(other.min(axis=1) / np.maximum(own, 1e-12))
    for name, values in out.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 점수에 비유한 값이 있습니다.")
    return out


def measure_latency(ref, emb, p1, requests=2000, seed=42):
    """요청 1건씩 점수만 계산한 CPU 시간의 중앙값(ms). 1차 표현·확률은 이미 있다고 본다(§8.14 ④)."""
    emb = _embeddings(emb)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(emb), requests)
    p1 = np.asarray(p1)
    # 점수 하나씩 따로 재야 점수별 비용 조건을 판정할 수 있다 → 단일 점수 경로를 분리한다.
    single = {
        "maha": lambda z, p: _class_mahalanobis(z, ref["means_t"], ref["chol"]).min(),
        "rmaha": lambda z, p: (_class_mahalanobis(z, ref["means_t"], ref["chol"]).min()
                               - _class_mahalanobis(z, ref["bg_mean_t"], ref["bg_chol"])[0, 0]),
        "knn": lambda z, p: np.partition(_bank_distances(_normalize(z), ref["bank"]),
                                         KNN_K - 1, axis=1)[0, KNN_K - 1],
        "trust": lambda z, p: _trust_single(ref, z, p),
    }
    result = {}
    for name, fn in single.items():
        timings = []
        for row in rows:
            z, p = emb[row:row + 1], p1[row:row + 1]
            started = time.perf_counter_ns()
            fn(z, p)
            timings.append(time.perf_counter_ns() - started)
        result[name] = float(np.median(timings) / 1e6)
    return result


def _trust_single(ref, z, p):
    nearest = _class_nearest(_bank_distances(_normalize(z), ref["bank"]), ref["bank_starts"])[0]
    pred = int(p.argmax())
    own = nearest[pred]
    nearest[pred] = np.inf
    return -(nearest.min() / max(own, 1e-12))
