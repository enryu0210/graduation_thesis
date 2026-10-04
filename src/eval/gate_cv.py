"""동일 분할의 CV 확률을 비교하여 게이트 과잉 승급 배수의 쌍체 검정을 수행한다."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import gate_signals
from src.eval import metrics
from src.eval.cv_compare import holm_bonferroni

SIGNALS = ("msp_T", "margin", "entropy")


def load_cvprobs(path):
    """pickle 없이 읽고 인덱스·확률 계약을 검사하여 잘못된 대응을 막는다."""
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    required = {"label_fingerprint", "y", "classes"}
    if not required <= data.keys():
        raise ValueError("CV 확률 파일에 지문·라벨·클래스가 필요합니다.")
    classes, y = data["classes"], data["y"]
    if (classes.ndim != 1 or len(classes) < 2 or classes.dtype.kind != "U"
            or len(set(classes)) != len(classes) or y.ndim != 1 or not len(y)
            or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= len(classes)).any()
            or data["label_fingerprint"].shape != ()):
        raise ValueError("CV 확률 파일의 클래스·라벨·지문 형식이 잘못되었습니다.")
    fold_keys = set(data) - required
    matches = [re.fullmatch(r"(?:val|test)_(?:idx|probs)_([1-9][0-9]*)", key) for key in fold_keys]
    if not matches or any(match is None for match in matches):
        raise ValueError("CV fold 키 형식이 잘못되었습니다.")
    folds = sorted({int(match.group(1)) for match in matches})
    expected = {f"{split}_{name}_{k}" for k in folds for split in ("val", "test")
                for name in ("idx", "probs")}
    if folds != list(range(1, len(folds) + 1)) or fold_keys != expected:
        raise ValueError("CV fold 번호 또는 필수 배열이 누락되었습니다.")
    for k in folds:
        for split in ("val", "test"):
            idx, probs = data[f"{split}_idx_{k}"], data[f"{split}_probs_{k}"]
            if (idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer)
                    or (idx < 0).any() or (idx >= len(y)).any() or len(np.unique(idx)) != len(idx)
                    or probs.shape != (len(idx), len(classes))):
                raise ValueError(f"fold {k} {split} 인덱스와 확률의 대응이 잘못되었습니다.")
            # 입력 검사도 기존 신호 함수를 사용해 확률 계약을 공유한다.
            gate_signals.gate_confidence(probs, "msp")
        if np.intersect1d(data[f"val_idx_{k}"], data[f"test_idx_{k}"]).size:
            raise ValueError(f"fold {k} val과 test가 겹칩니다.")
    return data, folds


def validate_alignment(stage1, stage2):
    """행 순서 하나가 달라도 쌍체 비교가 무효이므로 즉시 중단한다."""
    first, folds1 = load_cvprobs(stage1)
    second, folds2 = load_cvprobs(stage2)
    for key in ("label_fingerprint", "y", "classes"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"두 CV 파일의 {key}가 다릅니다.")
    if folds1 != folds2:
        raise ValueError("두 CV 파일의 fold 수가 다릅니다.")
    for k in folds1:
        for split in ("val", "test"):
            key = f"{split}_idx_{k}"
            if not np.array_equal(first[key], second[key]):
                raise ValueError(f"두 CV 파일의 {key}가 다릅니다.")
    return first, second, folds1


def fold_reports(first, second, folds, offset):
    """모든 fold에 같은 목표 여유를 적용하여 신호 간 비교 기준을 유지한다."""
    classes = first["classes"].tolist()
    reports = []
    for k in folds:
        y_val = first["y"][first[f"val_idx_{k}"]]
        y_test = first["y"][first[f"test_idx_{k}"]]
        p2_test = second[f"test_probs_{k}"]
        teacher = metrics.compute_metrics(y_test, p2_test.argmax(1), classes)["macro_f1"]
        target = teacher - offset
        report = gate_signals.signal_report(first[f"val_probs_{k}"], y_val,
                                            first[f"test_probs_{k}"], p2_test,
                                            y_test, len(classes), target)
        reports.append({"fold": k, "stage2_macro_f1": teacher,
                        "target_macro_f1": target, **report})
    return reports


def reports_with_fallback(first, second, folds, target_offset=0., fallback_offset=.005):
    """한 fold의 MSP 실패도 전체 fold·전체 신호의 재계산으로 처리한다."""
    if (not np.isfinite([target_offset, fallback_offset]).all()
            or not 0 <= target_offset <= fallback_offset <= 1):
        raise ValueError("offset은 0~1이고 fallback-offset은 target-offset 이상이어야 합니다.")
    reports = fold_reports(first, second, folds, target_offset)
    failed = [row["fold"] for row in reports
              if not row["signals"]["msp"]["over_escalation_ratio"]["reached"]]
    fallback = target_offset == 0 and bool(failed)
    if fallback:
        reports = fold_reports(first, second, folds, fallback_offset)
    return {"requested_offset": target_offset, "fallback_offset": fallback_offset,
            "used_offset": fallback_offset if fallback else target_offset,
            "fallback_applied": fallback, "initial_msp_unreached_folds": failed,
            "folds": reports}


def compare_ratios(reports, signals=SIGNALS):
    """결측을 0으로 취급하지 않고 세 검정의 Holm 모집단을 유지한다."""
    comparisons, pvalues = {}, []
    valid_names = []
    for signal in signals:
        pairs = [(row["signals"]["msp"]["over_escalation_ratio"]["ratio"],
                  row["signals"][signal]["over_escalation_ratio"]["ratio"])
                 for row in reports]
        missing = [reports[i]["fold"] for i, pair in enumerate(pairs)
                   if any(value is None or not np.isfinite(value) for value in pair)]
        item = {"mean_msp_ratio": None, "mean_signal_ratio": None, "reduction": None,
                "raw_p": None, "adjusted_p": None, "h_e5_1": None, "reason": None,
                "ratios": [{"fold": row["fold"], "msp": pair[0], "signal": pair[1]}
                           for row, pair in zip(reports, pairs)]}
        comparisons[signal] = item
        if missing:
            item["reason"] = f"ratio가 결측인 fold가 있어 검정을 생략합니다: {missing}"
            continue
        base, other = np.asarray(pairs, dtype=float).T
        item["mean_msp_ratio"], item["mean_signal_ratio"] = float(base.mean()), float(other.mean())
        if len(pairs) < 2 or base.mean() == 0:
            item["reason"] = "fold가 2개 미만이거나 MSP 평균이 0이라 검정·감소율을 정의할 수 없습니다."
            continue
        item["reduction"] = float(1 - other.mean() / base.mean())
        # 상수 차이는 scipy 경고가 발생할 수 있어 유한성 판정 후 별도 사유로 보존한다.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pvalue = float(ttest_rel(other, base).pvalue)
        if not np.isfinite(pvalue):
            item["reason"] = "쌍체 t-test p값이 비유한 값이라 검정을 판정하지 않습니다."
            continue
        item["raw_p"] = pvalue
        valid_names.append(signal)
        pvalues.append(pvalue)
    # 검정 불가 쌍은 결과에 p를 만들지 않되 보정 내부에서만 1을 넣어 3쌍을 유지한다.
    adjusted = holm_bonferroni(pvalues + [1.] * (len(signals) - len(pvalues)))
    for signal, pvalue in zip(valid_names, adjusted):
        item = comparisons[signal]
        item["adjusted_p"] = pvalue
        item["h_e5_1"] = bool(item["reduction"] >= .20 and pvalue < .05)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--target-offset", type=float, default=0.)
    parser.add_argument("--fallback-offset", type=float, default=.005)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        first, second, folds = validate_alignment(args.stage1, args.stage2)
        result = reports_with_fallback(first, second, folds, args.target_offset, args.fallback_offset)
        result.update({"stage1": args.stage1.name, "stage2": args.stage2.name,
                       "classes": first["classes"].tolist(),
                       "label_fingerprint": first["label_fingerprint"].item(),
                       "comparisons": compare_ratios(result["folds"]), "holm_family_size": 3})
        output = args.out or (PROJECT_ROOT / "experiments/results" /
                             f"gate_cv_{args.stage1.stem}_vs_{args.stage2.stem}.json")
        content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(content, end="")
    except (OSError, ValueError, KeyError) as error:
        print(f"CV 게이트 비교 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
