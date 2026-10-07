"""경계(frontier)는 '그 속도 이상에서의 최고 F1' 이어야 한다 — 예산 순서가 아니라 속도 기준으로 비교한다."""
import numpy as np

from src.eval.gate_frontier import frontier, speedup


def test_frontier_takes_best_f1_at_or_above_each_speed():
    speeds = [1.0, 2.0, 1.5, 3.0]          # G2 탐욕 경로처럼 순서가 섞여 있다
    f1s = [0.99, 0.95, 0.90, 0.80]
    got = frontier(speeds, f1s, [1.0, 1.6, 2.5, 3.5])
    assert np.allclose(got[:3], [0.99, 0.95, 0.80]) and np.isnan(got[3])   # 1.5배 점(0.90)은 2.0배 점에 지배된다


def test_speedup_ends_match_single_models():
    assert speedup(0, 0, 0.35, 1.114) == 1.114 / 0.35          # 넘김 0% → 1차 단독
    assert np.isclose(speedup(1, 1, 0.35, 1.114), 1.114 / 1.464)  # 전량 넘김 → 두 모델 합
