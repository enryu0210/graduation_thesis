"""E8 판정 규칙 — 사전 문서 §E8 의 세 조건이 그대로 옮겨졌는지 고정한다."""
from src.eval import dl_cascade as D


def row(folds, f1):
    return {"b0": {"macro_f1": f1}, "per_fold_speedup": {"b0": folds}}


def test_judge_requires_all_three_prereg_conditions():
    assert D.judge(row([1.3] * 5, 0.988), 0.9884)["above"]
    assert not D.judge(row([1.1] * 5, 0.988), 0.9884)["above"]              # 평균 1.2 미달
    assert not D.judge(row([2.0, 2.0, 2.0, 2.0, 0.9], 0.988), 0.9884)["above"]  # 한 fold 가 1 이하
    assert not D.judge(row([1.3] * 5, 0.9850), 0.9884)["above"]             # F1 0.34pp 손실
    assert not D.judge({"reason": "점 없음"}, 0.9884)["above"]


def test_cvprobs_suffix_follows_cross_validate_tag_rule():
    assert D.cvprobs_suffix(42) == "" and D.cvprobs_suffix(42, aux=True) == "_aux"
    assert D.cvprobs_suffix(43) == "_s43" and D.cvprobs_suffix(43, aux=True) == "_s43_aux"


def test_k3_alone_applies_e8_conditions_without_deferral():
    import numpy as np
    y = np.array([0, 1, 2, 3] * 5)
    def probs(correct):
        out = {"y": y}
        for k in range(1, 6):
            p = np.eye(4)[y if correct else (y + 1) % 4]
            out[f"test_idx_{k}"], out[f"test_probs_{k}"] = np.arange(len(y)), p
        return out
    good = D.k3_alone(probs(True), probs(True), 0.9884, t2=0.345, t_front=0.095)
    assert good["e9_1"]["above"] and good["macro_f1"] == 1.0 and good["dl_c_aux_minus_k3_pp"] == [0.0] * 5
    assert not D.k3_alone(probs(False), probs(True), 0.9884, t2=0.345, t_front=0.095)["e9_1"]["above"]


def test_e10_2_above_r_needs_speed_and_f1():
    # 사전 문서 §E10-2: t_R / t_X ≥ 1.2 그리고 F1_X ≥ F1_R − 0.3%p.
    from src.eval import dl_a
    r = [0.990, 0.991, 0.989, 0.990, 0.990]
    assert dl_a.above_r(12.0, 10.0, [0.988] * 5, r)["above"]
    assert not dl_a.above_r(11.9, 10.0, [0.990] * 5, r)["above"]        # 1.19배
    assert not dl_a.above_r(20.0, 1.0, [0.9865] * 5, r)["above"]        # 0.35%p 손실
