"""CV 확률만으로 클래스별 임계값의 과잉 넘김과 배포 비용을 비교한다."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.gate_cv import validate_alignment
from src.eval.metrics import compute_metrics
from src.models import gate_signals as gates


def deployment_cost(r_normal, r_attack, prevalence, ms1, ms2):
    """실제 라벨별 넘김률로 공격 비율 변화에 따른 지연을 근사한다."""
    if (not np.isfinite([r_normal, r_attack, prevalence, ms1, ms2]).all()
            or not all(0 <= v <= 1 for v in (r_normal, r_attack, prevalence))
            or ms1 <= 0 or ms2 <= 0):
        raise ValueError("넘김률·공격 비율은 [0,1], 지연은 유한 양수여야 합니다.")
    rate = prevalence * r_attack + (1 - prevalence) * r_normal
    return {"prevalence": prevalence, "escalation_rate": rate,
            "speedup": ms2 / (ms1 + rate * ms2)}


def gate_report(curve, p1_test, y_test, normal_index, target, val_target, oracle, ms1, ms2):
    """목표 도달점과 val에서 고른 운영점을 분리해 test 선택 편향을 명시한다."""
    reached = [row["test_escalation_rate"] for row in curve if row["test_macro_f1"] >= target]
    needed = min(reached) if reached else None
    eligible = [row for row in curve if row["val_macro_f1"] >= val_target]
    operating = None
    if eligible:
        chosen = min(eligible, key=lambda row: row["val_escalation_rate"])
        conf = p1_test.max(axis=1)
        mask = (gates.apply_classwise(p1_test, conf, chosen["taus"]) if "taus" in chosen
                else conf < chosen["tau"])
        normal = y_test == normal_index
        # 실제 라벨층이 없으면 재가중을 정의할 수 없으므로 값을 꾸며내지 않는다.
        if not normal.any() or normal.all():
            raise ValueError("운영점 재가중에는 test 정상·공격 행이 모두 필요합니다.")
        r_normal, r_attack = float(mask[normal].mean()), float(mask[~normal].mean())
        prevalence = float((~normal).mean())
        operating = {**chosen, "r_N": r_normal, "r_A": r_attack,
                     "test_attack_prevalence": prevalence,
                     "test_minus_val_escalation": chosen["test_escalation_rate"] - chosen["val_escalation_rate"],
                     "deployment": {name: deployment_cost(r_normal, r_attack, pi, ms1, ms2)
                                    for name, pi in (("fold_actual", prevalence), ("pi_0.033", .033),
                                                     ("pi_0.001", .001))}}
    return {"curve": curve, "needed_test_escalation": needed,
            "reached": needed is not None, "ratio": needed / oracle if needed is not None and oracle else None,
            "operating_point": operating}


def fold_curves(first, second, folds, step):
    """임계값 경로는 val만 보고 생성하며 목표 하향에도 동일 경로를 재사용한다."""
    classes = first["classes"].tolist()
    if "Normal" not in classes:
        raise ValueError("classes에 Normal이 필요합니다.")
    reports = []
    for k in folds:
        y_val, y_test = [first["y"][first[f"{split}_idx_{k}"]] for split in ("val", "test")]
        p1_val, p1_test = [first[f"{split}_probs_{k}"] for split in ("val", "test")]
        p2_val, p2_test = [second[f"{split}_probs_{k}"] for split in ("val", "test")]
        teacher_val = compute_metrics(y_val, p2_val.argmax(1), classes)["macro_f1"]
        teacher_test = compute_metrics(y_test, p2_test.argmax(1), classes)["macro_f1"]
        conf_val, conf_test = p1_val.max(1), p1_test.max(1)
        b0 = []
        for budget in np.r_[np.arange(0, 1, step), 1.0]:
            tau = gates.tau_for_budget(conf_val, budget)
            pred_val, _, mask_val = gates.cascade_apply_signal(p1_val, p2_val, conf_val, tau)
            pred_test, _, mask_test = gates.cascade_apply_signal(p1_test, p2_test, conf_test, tau)
            b0.append({"budget": float(budget), "tau": tau,
                       "val_escalation_rate": float(mask_val.mean()),
                       "val_macro_f1": gates._macro_f1_fast(y_val, pred_val, len(classes)),
                       "test_escalation_rate": float(mask_test.mean()),
                       "test_macro_f1": gates._macro_f1_fast(y_test, pred_test, len(classes))})
        g2 = gates.classwise_tau_path(p1_val, p2_val, y_val, len(classes), step)
        for point in g2:
            mask = gates.apply_classwise(p1_test, conf_test, point["taus"])
            pred = np.where(mask, p2_test.argmax(1), p1_test.argmax(1))
            point.update({"test_escalation_rate": float(mask.mean()),
                          "test_macro_f1": gates._macro_f1_fast(y_test, pred, len(classes))})
        reports.append({"fold": k, "stage2_val_macro_f1": teacher_val,
                        "stage2_test_macro_f1": teacher_test,
                        "oracle": float((p1_test.argmax(1) != y_test).mean()),
                        "curves": {"b0": b0, "g2": g2}})
    return reports


def reports_with_fallback(first, second, folds, step=.005, target_offset=0., fallback_offset=.005,
                          ms1=.350, ms2=1.114):
    """B0 한 fold의 미도달도 모든 fold·두 게이트의 목표를 함께 낮춘다."""
    if (not np.isfinite([step, target_offset, fallback_offset]).all() or not 0 < step <= 1
            or not 0 <= target_offset <= fallback_offset <= 1):
        raise ValueError("step은 (0,1], offset은 0 <= target <= fallback <= 1이어야 합니다.")
    deployment_cost(0., 0., 0., ms1, ms2)
    reports = fold_curves(first, second, folds, step)
    failed = [row["fold"] for row in reports if not any(
        point["test_macro_f1"] >= row["stage2_test_macro_f1"] - target_offset
        for point in row["curves"]["b0"])]
    fallback = target_offset == 0 and bool(failed)
    offset = fallback_offset if fallback else target_offset
    normal_index = first["classes"].tolist().index("Normal")
    for row in reports:
        k = row["fold"]
        target = row["stage2_test_macro_f1"] - offset
        row["target_macro_f1"] = target
        row["gates"] = {name: gate_report(curve, first[f"test_probs_{k}"],
                           first["y"][first[f"test_idx_{k}"]], normal_index, target,
                           row["stage2_val_macro_f1"] - .005, row["oracle"], ms1, ms2)
                        for name, curve in row.pop("curves").items()}
    return {"requested_offset": target_offset, "fallback_offset": fallback_offset,
            "used_offset": offset, "fallback_applied": fallback,
            "initial_b0_unreached_folds": failed, "folds": reports}


def summarize(reports, classes):
    """결측 fold를 0으로 대체하지 않고 검정 불가 사유를 남긴다."""
    summary = {"folds": [{"fold": row["fold"], "oracle": row["oracle"],
                          **{f"{name}_{key}": row["gates"][name][source]
                             for name in ("b0", "g2")
                             for key, source in (("needed", "needed_test_escalation"), ("ratio", "ratio"))}}
                         for row in reports],
               "mean_b0_ratio": None, "mean_g2_ratio": None, "reduction": None,
               "p": None, "h_g2": None, "reason": None, "operating_points": {}}
    pairs = [(row["b0_ratio"], row["g2_ratio"]) for row in summary["folds"]]
    missing = [row["fold"] for row, pair in zip(summary["folds"], pairs) if None in pair]
    if missing:
        summary["reason"] = f"ratio 결측 fold가 있어 검정을 생략합니다: {missing}"
    else:
        b0, g2 = np.asarray(pairs).T
        summary.update(mean_b0_ratio=float(b0.mean()), mean_g2_ratio=float(g2.mean()))
        if b0.mean() == 0 or len(pairs) < 2:
            summary["reason"] = "B0 평균이 0이거나 fold가 2개 미만이어서 검정할 수 없습니다."
        else:
            summary["reduction"] = float(1 - g2.mean() / b0.mean())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = float(ttest_rel(g2, b0).pvalue)
            if np.isfinite(p):
                summary.update(p=p, h_g2=bool(summary["reduction"] >= .20 and p < .05))
            else:
                summary["reason"] = "paired t-test p가 비유한 값이어서 판정을 생략합니다."
    for name in ("b0", "g2"):
        points = [row["gates"][name]["operating_point"] for row in reports]
        if any(point is None for point in points):
            summary["operating_points"][name] = {"reason": "val 목표 미도달 운영점이 있습니다."}
            continue
        item = {f"mean_{key}": float(np.mean([point[key] for point in points]))
                for key in ("test_escalation_rate", "test_macro_f1", "r_N", "r_A",
                            "test_minus_val_escalation")}
        item["mean_speedup"] = {pi: float(np.mean([point["deployment"][pi]["speedup"] for point in points]))
                                for pi in ("fold_actual", "pi_0.033", "pi_0.001")}
        if name == "g2":
            taus = np.array([point["taus"] for point in points])
            item["taus_by_class"] = {c: {"mean": float(taus[:, i].mean()),
                                          "std": float(taus[:, i].std(ddof=0))}
                                     for i, c in enumerate(classes)}
            item["tau_std_ddof"] = 0
        summary["operating_points"][name] = item
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    for name, default in (("step", .005), ("target-offset", 0.), ("fallback-offset", .005),
                          ("ms1", .350), ("ms2", 1.114)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        first, second, folds = validate_alignment(args.stage1, args.stage2)
        result = reports_with_fallback(first, second, folds, args.step, args.target_offset,
                                       args.fallback_offset, args.ms1, args.ms2)
        classes = first["classes"].tolist()
        result.update(stage1=args.stage1.name, stage2=args.stage2.name, classes=classes,
                      label_fingerprint=first["label_fingerprint"].item(), step=args.step,
                      ms1=args.ms1, ms2=args.ms2, latency_note="T2 단일 split의 CPU·배치1 지연을 사용한 근사",
                      summary=summarize(result["folds"], classes))
        output = args.out or PROJECT_ROOT / "experiments/results" / f"gate_g2_{args.stage1.stem}_vs_{args.stage2.stem}.json"
        # 기존 연구 산출물을 실수로 덮어쓰지 않도록 배타적 생성한다.
        output.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with output.open("x", encoding="utf-8") as saved:
            saved.write(content)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(f"G2 게이트 비교 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
