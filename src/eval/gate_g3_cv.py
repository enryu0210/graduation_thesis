"""CV의 동일 fold에서 B0·G3 라우터 과잉 넘김과 배포 비용을 비교한다."""
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
from src.eval import cross_validate, gate_cv, gate_g2, gate_g3

GATES = ("b0", "g3_lite", "g3_full")


def load_embeddings(path, first, folds):
    """동일 라벨의 행 교환도 막도록 모든 풀 인덱스를 대조한다."""
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    expected = {"classes", "y", "label_fingerprint"} | {
        f"{split}_{kind}_{k}" for k in folds for split in ("val", "test")
        for kind in ("idx", "emb")}
    if set(data) != expected:
        raise ValueError("CV 표현 파일의 fold 키가 일치하지 않습니다.")
    for key in ("classes", "y", "label_fingerprint"):
        if not np.array_equal(data[key], first[key]):
            raise ValueError(f"CV 표현 파일의 {key}가 다릅니다.")
    for k in folds:
        for split in ("val", "test"):
            idx = f"{split}_idx_{k}"
            if not np.array_equal(data[idx], first[idx]):
                raise ValueError(f"CV 표현 파일의 {idx}가 다릅니다.")
            emb = data[f"{split}_emb_{k}"]
            if emb.shape != (len(first[idx]), 128) or not np.isfinite(emb).all():
                raise ValueError(f"fold {k} {split} 표현은 유한한 N×128 배열이어야 합니다.")
    return data


def summarize(reports):
    """결측 배율을 제외한 유리한 fold만 검정하지 않도록 결측 사유를 남긴다."""
    summary = {"folds": [{"fold": row["fold"], "oracle": row["oracle"],
                           **{f"{name}_ratio": row["gates"][name]["ratio"] for name in GATES}}
                          for row in reports], "operating_points": {}, "comparisons": {}}
    for name in GATES:
        ratios = [row["gates"][name]["ratio"] for row in reports]
        summary[f"mean_{name}_ratio"] = (float(np.mean(ratios))
                                           if all(r is not None for r in ratios) else None)
        points = [row["gates"][name]["operating_point"] for row in reports]
        item = {"router_latency_included": name == "b0"}
        if any(point is None for point in points):
            item["reason"] = "val 목표 미도달 운영점이 있습니다."
        else:
            item.update({f"mean_{key}": float(np.mean([point[key] for point in points]))
                         for key in ("test_escalation_rate", "test_macro_f1", "r_N", "r_A")})
            item["mean_speedup_pi_0.033"] = float(np.mean([
                point["deployment"]["pi_0.033"]["speedup"] for point in points]))
        if name != "b0":
            item["latency_note"] = "라우터 지연 미포함; T2 CPU·배치1 0.350/1.114 ms 근사"
        summary["operating_points"][name] = item
    for name, hypothesis in (("g3_lite", "H-G3L-cv"), ("g3_full", "H-G3F-cv")):
        item = {"exploratory": name == "g3_lite", "reduction": None,
                "p": None, "hypothesis": hypothesis, "adopted": None, "reason": None}
        base = [row["gates"]["b0"]["ratio"] for row in reports]
        other = [row["gates"][name]["ratio"] for row in reports]
        if any(value is None for value in base + other):
            item["reason"] = "ratio 결측 fold가 있어 검정을 생략합니다."
        elif len(base) < 2 or np.mean(base) == 0:
            item["reason"] = "B0 평균이 0이거나 fold가 2개 미만입니다."
        else:
            item["reduction"] = float(1 - np.mean(other) / np.mean(base))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = float(ttest_rel(other, base).pvalue)
            if np.isfinite(p):
                item.update(p=p, adopted=bool(item["reduction"] >= .20 and p < .05))
            else:
                item["reason"] = "paired t-test p가 비유한 값이어서 판정을 생략합니다."
        summary["comparisons"][name] = item
    full = summary["comparisons"]["g3_full"]
    summary.update(reduction=full["reduction"], p=full["p"], **{"H-G3F-cv": full["adopted"]})
    return summary


def analyze(args):
    """기존 라우터 규칙을 재사용하며 test는 적합과 임계값 선택에 쓰지 않는다."""
    if not np.isfinite(args.step) or not 0 < args.step <= 1:
        raise ValueError("step은 (0,1]이어야 합니다.")
    first, second, folds = gate_cv.validate_alignment(args.stage1, args.stage2)
    emb = load_embeddings(args.embeddings, first, folds)
    # load_text_pool은 CSV 열 이름 text_raw 대신 표현 선택자 raw를 받는다.
    texts, labels, classes = cross_validate.load_text_pool(args.track, "raw", exclude_test=True)
    if (not np.array_equal(labels, first["y"]) or classes != first["classes"].tolist()
            or len(texts) != len(labels)):
        raise ValueError("원문 풀과 CV 확률의 라벨·클래스 정렬 불일치")
    features = gate_g3.text_features(texts)
    reports = gate_g2.fold_curves(first, second, folds, args.step)
    failed = [row["fold"] for row in reports if not any(
        point["test_macro_f1"] >= row["stage2_test_macro_f1"] for point in row["curves"]["b0"])]
    offset = .005 if failed else 0.
    for row in reports:
        k = row["fold"]
        vi, ti = first[f"val_idx_{k}"], first[f"test_idx_{k}"]
        yv, yt = first["y"][vi], first["y"][ti]
        p1v, p1t = first[f"val_probs_{k}"], first[f"test_probs_{k}"]
        p2v, p2t = second[f"val_probs_{k}"], second[f"test_probs_{k}"]
        target = (p1v.argmax(1) != yv) & (p2v.argmax(1) == yv)
        curves = {"b0": row.pop("curves")["b0"]}
        confidences = {"b0": p1t.max(1)}
        for name in GATES[1:]:
            ev, et = ((emb[f"val_emb_{k}"], emb[f"test_emb_{k}"])
                      if name == "g3_full" else (None, None))
            xv = gate_g3.router_features(p1v, features[vi], ev)
            xt = gate_g3.router_features(p1t, features[ti], et)
            sv, st, _, _ = gate_g3.cross_fit_router(xv, yv, target, xt, args.seed)
            curves[name] = gate_g3.router_curve(p1v, p2v, yv, p1t, p2t, yt, sv, st, args.step)
            confidences[name] = 1 - st
        row["target_macro_f1"] = row["stage2_test_macro_f1"] - offset
        row["gates"] = {}
        for name, curve in curves.items():
            reached = [point["test_escalation_rate"] for point in curve
                       if point["test_macro_f1"] >= row["target_macro_f1"]]
            needed = min(reached) if reached else None
            eligible = [point for point in curve
                        if point["val_macro_f1"] >= row["stage2_val_macro_f1"] - .005]
            operating = None
            if eligible:
                chosen = min(eligible, key=lambda point: point["val_escalation_rate"])
                mask = confidences[name] < chosen["tau"]
                operating = gate_g3.operating_summary(chosen, mask, yt, classes.index("Normal"), .350, 1.114)
            row["gates"][name] = {"curve": curve, "needed_test_escalation": needed,
                                   "ratio": needed / row["oracle"] if needed is not None and row["oracle"] else None,
                                   "reached": needed is not None, "operating_point": operating}
    return {"stage1": args.stage1.name, "stage2": args.stage2.name,
            "embeddings": args.embeddings.name, "track": args.track, "classes": classes,
            "label_fingerprint": first["label_fingerprint"].item(), "step": args.step, "seed": args.seed,
            "requested_offset": 0., "fallback_offset": .005, "used_offset": offset,
            "fallback_applied": bool(failed), "initial_b0_unreached_folds": failed,
            "latency_note": "T2 CPU·배치1 0.350/1.114 ms; G3 라우터 지연 미포함",
            "folds": reports, "summary": summarize(reports)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1", "stage2", "embeddings"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--track", default="srbh_4class", choices=("srbh_4class",))
    parser.add_argument("--step", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed는 0 이상이어야 합니다.")
    try:
        output = args.out or PROJECT_ROOT / f"experiments/results/gate_g3_cv_{args.stage1.stem}.json"
        if output.exists():
            raise FileExistsError("기존 CV 게이트 결과를 덮어쓰지 않습니다.")
        result = analyze(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as saved:
            saved.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"G3 CV 비교 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
