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
