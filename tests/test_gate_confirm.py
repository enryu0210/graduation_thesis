"""C-G2 판정 규칙이 사전 기준 문서(03_prereg_v2.md)의 문장과 같은 계산인지 고정한다."""
import numpy as np

from src.eval import gate_confirm as GC


def _path(points):
    """(E_π, test F1, val F1) 목록 → fold_paths 한 fold 의 7열 행렬. 정상·공격 넘김률을 같게 둬 E_π 가 그대로 나온다."""
    return np.array([(e, e, e, f1, e, e, vf1) for e, f1, vf1 in points])


def test_corrected_t_is_wider_than_plain_paired_t():
    d = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
    got = GC.corrected_t(d)
    plain = d.mean() / np.sqrt(d.var(ddof=1) / 5)
    assert np.isclose(got["t"], plain * np.sqrt(0.2 / 0.45))     # (1/5) → (1/5 + 1/4)
    assert GC.corrected_t([0.1] * 5) == {"t": None, "p": None}   # 분산 0 은 검정 불가


def test_primary_rule_needs_both_conditions():
    b0 = _path([(0.0, 0.90, 0.9), (0.2, 0.95, 0.9), (0.5, 0.97, 0.9)])
    better = _path([(0.0, 0.91, 0.9), (0.18, 0.96, 0.9), (0.3, 0.975, 0.9)])
    passed = GC.primary_rule({"b0": [b0] * 5, "g2": [better] * 5}, GC.R_ALL)
    assert passed["passed"] and passed["points"]["0.186"]["folds_above"] == 5
    assert np.isclose(passed["points"]["0.353"]["mean_pp"], 2.5)   # 0.975 − 0.95
    # 평균은 위지만 G2 가 위인 fold 가 3/5 뿐이면 (ii) 에서 떨어진다.
    worse = _path([(0.0, 0.89, 0.9), (0.18, 0.949, 0.9), (0.3, 0.949, 0.9)])
    mixed = GC.primary_rule({"b0": [b0] * 5, "g2": [better] * 3 + [worse] * 2}, GC.R_ALL)
    assert mixed["rule_i_all_points_positive"] and not mixed["rule_ii_folds"] and not mixed["passed"]


def test_secondary_rule_picks_the_point_on_val_not_test():
    # val 목표(0.95 − 0.005)를 넘는 점 중 넘김이 가장 적은 점을 고른다 — test F1 이 더 높은 점이 있어도 무시.
    b0 = _path([(0.0, 0.90, 0.90), (0.4, 0.96, 0.95), (0.6, 0.99, 0.96)])
    g2 = _path([(0.0, 0.90, 0.90), (0.2, 0.96, 0.95), (0.6, 0.99, 0.96)])
    got = GC.secondary_rule({"b0": [b0] * 5, "g2": [g2] * 5}, [0.95] * 5, GC.R_ALL, 1.0, 4.0)
    assert np.isclose(got["b0"]["e_pi"], 0.4) and np.isclose(got["g2"]["e_pi"], 0.2)
    assert np.isclose(got["speedup_gain"], (4 / 1.8) / (4 / 2.6) - 1)
    assert got["adopted"] is False                                # fold 차이의 분산이 0 → p 없음 → 채택 불가
    assert "reason" in GC.secondary_rule({"b0": [b0] * 5, "g2": [g2] * 5}, [0.99] * 5, GC.R_ALL, 1.0, 4.0)
