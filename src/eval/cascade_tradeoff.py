"""val에서 고정한 승급 예산으로 정확도와 순전파 비용의 교환을 측정한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import cascade, gate_signals
from src.eval import metrics
from src.analysis import srbh_normal_strata

BUDGETS = (0, .01, .02, .05, .10, .20, .30, .50, 1.0)


def select_operating_point(rows, teacher_val_f1, tolerance=.005):
    """test 성능을 선택에 쓰지 않아 운영점 선택 누수를 막는다."""
    feasible = [row for row in rows
                if row["val_macro_f1"] >= teacher_val_f1 - tolerance]
    return min(feasible, key=lambda row: row["budget"]) if feasible else None


def reweighted_escalation(attack_rate, normal_rate, prevalence):
    """배포 환경의 공격 비율만 바꾸고 클래스 조건부 승급률은 유지한다."""
    values = np.asarray([attack_rate, normal_rate, prevalence], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("승급률과 공격 비율은 0~1 사이의 유한한 값이어야 합니다.")
    return float(prevalence * attack_rate + (1 - prevalence) * normal_rate)


def latency_point(ms1, ms2, escalation_rate):
    """모든 요청의 1차 비용과 승급 요청의 2차 비용을 더한다."""
    if (not np.isfinite([ms1, ms2, escalation_rate]).all()
            or ms1 <= 0 or ms2 <= 0 or not 0 <= escalation_rate <= 1):
        raise ValueError("지연은 양수이고 승급률은 0~1이어야 합니다.")
    latency = ms1 + escalation_rate * ms2
    return {"cascade_ms": float(latency), "speedup": float(ms2 / latency),
            "escalation_rate": float(escalation_rate)}


def align_normal_strata(test_frame, strata_frame, y_test, classes):
    """층 CSV의 저장 순서를 믿지 않고 원래 test 행 순서로 재색인한다."""
    if not {"row_id", "label"} <= set(test_frame.columns):
        raise ValueError("test CSV에 row_id와 label이 필요합니다.")
    if not {"split", "row_id", "normal_source"} <= set(strata_frame.columns):
        raise ValueError("층 CSV에 split, row_id, normal_source가 필요합니다.")
    names = np.asarray(classes)[y_test]
    if len(test_frame) != len(y_test) or not np.array_equal(test_frame.label.to_numpy(), names):
        raise ValueError("test CSV 행 수 또는 라벨 순서가 추론 입력과 다릅니다.")
    selected = strata_frame.loc[strata_frame.split.eq("test")].copy()
    test_ids = test_frame.row_id.astype(str)
    source_ids = selected.row_id.astype(str)
    if (test_frame.row_id.isna().any() or selected.row_id.isna().any()
            or test_ids.duplicated().any() or source_ids.duplicated().any()):
        raise ValueError("row_id가 누락되거나 중복되었습니다.")
    normal_ids = test_ids[test_frame.label.eq("Normal")]
    if len(selected) != len(normal_ids) or set(source_ids) != set(normal_ids):
        raise ValueError("층 CSV의 test Normal 행 수 또는 row_id가 일치하지 않습니다.")
    mapping = pd.Series(selected.normal_source.to_numpy(), index=source_ids)
    aligned = test_ids.map(mapping).to_numpy()
    if not np.isin(aligned[names == "Normal"], srbh_normal_strata.STRATA).all():
        raise ValueError("Normal 행에 출처 층이 없거나 알 수 없는 층이 있습니다.")
    return aligned


def load_strata(track, limit, y_test, classes):
    """부분 표본이나 다른 트랙에 SR-BH 출처를 잘못 붙이지 않는다."""
    if limit is not None:
        return None, "--limit 부분 표본에서는 층별 FPR을 생략합니다."
    if track != "srbh_4class":
        return None, "SR-BH 이외의 트랙에는 SR-BH 출처 층을 적용하지 않습니다."
    path = PROJECT_ROOT / "data/processed/srbh_4class_normal_strata.csv"
    if not path.exists():
        raise FileNotFoundError("층 CSV가 없습니다. python src/analysis/srbh_normal_strata.py 를 실행하세요.")
    test = pd.read_csv(cascade.data_text.csv_path(track, "test"), dtype={"row_id": str})
    strata = pd.read_csv(path, dtype={"row_id": str})
    return align_normal_strata(test, strata, y_test, classes), None


def evaluate(y, probs, classes, strata=None):
    """논문 지표는 공통 metrics 모듈에 맡겨 계산 방식 차이를 막는다."""
    pred = probs.argmax(axis=1)
    report = metrics.compute_metrics(y, pred, classes)
    normal_idx = classes.index("Normal")
    attack = metrics.attack_focused_metrics(y, pred, classes, normal_idx)
    operating = metrics.attack_operating_points(y, probs, normal_idx)
    tpr = next(row["tpr"] for row in operating["tpr_at_fpr"] if row["target_fpr"] == .01)
    return {"macro_f1": report["macro_f1"], "accuracy": report["accuracy"],
            "tpr_at_1pct_fpr": tpr,
            "normal_fpr": attack["normal_false_positive_rate"],
            "fpr_by_stratum": (srbh_normal_strata.fpr_by_stratum(
                np.asarray(classes)[y], np.asarray(classes)[pred], strata)
                if strata is not None else None)}


def budget_table(p1_val, p2_val, y_val, p1_test, p2_test, y_test, classes, strata):
    """임계값은 val에서만 만들고 두 분할에 같은 미만 규칙을 적용한다."""
    val_msp = p1_val.max(axis=1)
    rows = []
    for budget in BUDGETS:
        tau = gate_signals.tau_for_budget(val_msp, budget)
        _, val_probs, val_mask = cascade.cascade_apply(p1_val, p2_val, tau)
        _, test_probs, test_mask = cascade.cascade_apply(p1_test, p2_test, tau)
        val = metrics.compute_metrics(y_val, val_probs.argmax(axis=1), classes)
        test = evaluate(y_test, test_probs, classes, strata)
        rows.append({"budget": budget, "tau": tau,
                     "val_escalation_rate": float(val_mask.mean()),
                     "test_escalation_rate": float(test_mask.mean()),
                     "val_macro_f1": val["macro_f1"],
                     **{f"test_{key}": value for key, value in test.items()},
                     "test_escalation_by_class": {
                         name: float(test_mask[y_test == k].mean()) if (y_test == k).any() else None
                         for k, name in enumerate(classes)}})
    return rows


def prevalence_table(op, p1_test, y_test, classes):
    """조건부 표본이 없으면 재가중을 정의할 수 없으므로 사유를 남긴다."""
    if op is None:
        return None, "val 조건을 만족하는 대표 운영점이 없습니다."
    mask = p1_test.max(axis=1) < op["tau"]
    attack = y_test != classes.index("Normal")
    if not attack.any() or attack.all():
        return None, "공격 또는 Normal 표본이 없어 조건부 승급률을 계산할 수 없습니다."
    r_a, r_n = float(mask[attack].mean()), float(mask[~attack].mean())
    prevalences = (.001, .01, .033, float(attack.mean()))
    return {"r_A": r_a, "r_N": r_n, "rows": [
        {"prevalence": pi, "escalation_rate": reweighted_escalation(r_a, r_n, pi)}
        for pi in prevalences]}, None


def measure_devices(net1, net2, x_img, x_seq, devices, batches, seed, op, reweighted):
    """CPU를 마지막에 측정하여 GPU로 모델을 되돌리는 복사를 피한다."""
    rows, skipped = [], []
    for name in sorted(devices, key=lambda name: name == "cpu"):
        if name == "cuda" and not torch.cuda.is_available():
            skipped.append({"device": name, "reason": "CUDA를 사용할 수 없습니다."})
            continue
        device = torch.device(name)
        net1.to(device)
        net2.to(device)
        for batch in batches:
            # 두 모델과 장치에 같은 샘플을 써 입력 구성 차이를 제거한다.
            indices = np.random.default_rng(seed).choice(len(x_img), batch, replace=batch > len(x_img))
            ms1 = cascade.measure_latency(net1, x_img, device, batch, sample_idx=indices)
            ms2 = cascade.measure_latency(net2, x_seq, device, batch, sample_idx=indices)
            row = {"device": name, "batch": batch, "ms1": ms1, "ms2": ms2,
                   "num_threads": torch.get_num_threads(),
                   "device_name": torch.cuda.get_device_name(device) if name == "cuda" else "CPU",
                   "cascade_ms": None, "speedup": None, "escalation_rate": None,
                   "reweighted": None}
            if op is not None:
                row.update(latency_point(ms1, ms2, op["test_escalation_rate"]))
            if reweighted is not None:
                row["reweighted"] = [{"prevalence": item["prevalence"],
                                      **latency_point(ms1, ms2, item["escalation_rate"])}
                                     for item in reweighted["rows"]]
            rows.append(row)
    return rows, skipped


def result_tag(args, encoders):
    """체크포인트 축과 표본·측정 축을 모두 반영하여 다른 실행의 덮어쓰기를 막는다."""
    base = cascade.checkpoint_tag(args.track, "cnn", args.text, args.balance,
                                  args.channels, encoders)
    limit = "all" if args.limit is None else str(args.limit)
    return (f"{base}_len{args.max_len}_n{limit}_s{args.seed}"
            f"_d{'-'.join(args.devices)}_b{'-'.join(map(str, args.batches))}")


def save_figure(result, path):
    """그림 글자는 ASCII로 한정하여 기본 폰트의 한글 깨짐을 피한다."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = result["budgets"]
    ax.plot([row["test_escalation_rate"] for row in rows],
            [row["test_macro_f1"] for row in rows], "o-", label="Cascade")
    for row in rows:
        ax.annotate(f"{row['budget']:.0%}", (row["test_escalation_rate"], row["test_macro_f1"]))
    for name, stage in result["stage_alone"].items():
        ax.axhline(stage["test"]["macro_f1"], ls="--", label=name)
    oracle = result["oracle"]
    ax.plot(oracle["escalation_rate"], oracle["macro_f1"], "*", ms=12, label="Oracle")
    ax.set(xlabel="Test escalation rate", ylabel="Test Macro-F1")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="srbh_4class")
    parser.add_argument("--channels", choices=("rgb", "gray"), default="rgb")
    parser.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy")
    parser.add_argument("--text", choices=("raw",), default="raw")
    parser.add_argument("--balance", action="store_true", help="균형 학습 체크포인트 사용(실행 시 명시)")
    parser.add_argument("--max-len", type=int, default=2304)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--devices", default="cuda,cpu")
    parser.add_argument("--batches", default="1,32,128,1024")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.devices = list(dict.fromkeys(name.strip() for name in args.devices.split(",")))
    try:
        args.batches = list(dict.fromkeys(int(value) for value in args.batches.split(",")))
    except ValueError:
        parser.error("--batches는 쉼표로 구분한 양의 정수여야 합니다.")
    if not args.devices or not set(args.devices) <= {"cuda", "cpu"}:
        parser.error("--devices는 cuda,cpu 중에서 지정하세요.")
    if not args.batches or min(args.batches) <= 0 or args.max_len <= 0 or args.seed < 0:
        parser.error("배치·최대 길이는 양수이고 seed는 0 이상이어야 합니다.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit는 양수여야 합니다.")
    if args.smoke and args.limit is None:
        args.limit = 500
    return args


def main():
    args = parse_args()
    try:
        encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
        device = torch.device("cuda" if "cuda" in args.devices and torch.cuda.is_available() else "cpu")
        print(f"입력 준비 및 기존 체크포인트 추론: {device}", flush=True)
        x_img_val, x_seq_val, y_val, classes = cascade.build_inputs(
            args.track, "val", args.text, 48, args.channels, encoders, args.max_len, args.limit)
        x_img_test, x_seq_test, y_test, test_classes = cascade.build_inputs(
            args.track, "test", args.text, 48, args.channels, encoders, args.max_len, args.limit)
        classes = list(classes)
        if classes != list(test_classes) or not len(y_val) or not len(y_test) or "Normal" not in classes:
            raise ValueError("분할별 클래스가 다르거나 빈 입력 또는 Normal 클래스가 없습니다.")
        strata, strata_reason = load_strata(args.track, args.limit, y_test, classes)
        net1 = cascade.load_net("cnn", len(classes), args.track, args.text, args.balance,
                                device, args.channels, 3 if args.channels == "rgb" else 1, encoders)
        net2 = cascade.load_net("charcnn", len(classes), args.track, args.text, args.balance, device)
        p1_val = cascade.predict_probs(net1, x_img_val, device, 512).astype(np.float32)
        p2_val = cascade.predict_probs(net2, x_seq_val, device, 512).astype(np.float32)
        p1_test = cascade.predict_probs(net1, x_img_test, device, 512).astype(np.float32)
        p2_test = cascade.predict_probs(net2, x_seq_test, device, 512).astype(np.float32)
        rows = budget_table(p1_val, p2_val, y_val, p1_test, p2_test, y_test, classes, strata)
        stage_alone = {name: {"val": evaluate(y_val, val, classes),
                              "test": evaluate(y_test, test, classes, strata)}
                       for name, val, test in (("RGB CNN" if args.channels == "rgb" else "gray CNN", p1_val, p1_test),
                                               ("char-CNN", p2_val, p2_test))}
        op = select_operating_point(rows, stage_alone["char-CNN"]["val"]["macro_f1"])
        reweighted, reweight_reason = prevalence_table(op, p1_test, y_test, classes)
        saturation = {split: {"fraction_equal_one": float((probs.max(1) == 1.0).mean()),
                              "msp_unique_values": int(len(np.unique(probs.max(1))))}
                      for split, probs in (("val", p1_val), ("test", p1_test))}
        print("saturation:", json.dumps(saturation, ensure_ascii=False), flush=True)
        print("예산 표:", json.dumps(rows, ensure_ascii=False), flush=True)
        print("장치별 지연 측정 시작", flush=True)
        latency, skipped = measure_devices(net1, net2, x_img_test, x_seq_test,
                                            args.devices, args.batches, args.seed, op, reweighted)
        headline = next((row for row in latency if row["device"] == "cpu" and row["batch"] == 1), None)
        for row in rows:
            row["cpu_batch1_latency"] = (latency_point(headline["ms1"], headline["ms2"],
                                                      row["test_escalation_rate"]) if headline else None)
        reference = min(row["cpu_batch1_latency"]["cascade_ms"] for row in rows) if headline else None
        for row in rows:
            row["fe_score"] = metrics.fe_score(row["test_macro_f1"], row["cpu_batch1_latency"]["cascade_ms"],
                                                reference_latency_ms=reference) if headline else None
        result = {"config": vars(args), "classes": classes, "saturation": saturation,
                  "budgets": rows, "stage_alone": stage_alone,
                  "oracle": cascade.oracle_point(p1_test, p2_test, y_test, classes),
                  "operating_point": op, "reweighted": reweighted, "reweight_reason": reweight_reason,
                  "strata_reason": strata_reason, "latency": latency, "skipped_devices": skipped,
                  "headline": headline, "fe_reference_latency_ms": reference,
                  "headline_reason": None if headline else "CPU·batch 1을 요청하지 않아 헤드라인과 FE를 생략합니다.",
                  "latency_note": "batch>1의 ms1+r·ms2는 근사입니다. 전처리 제외 순전파 지연입니다."}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.smoke:
            print("스모크: JSON·확률·그림 저장 생략")
            return 0
        tag = result_tag(args, encoders)
        output = PROJECT_ROOT / "experiments/results"
        output.mkdir(parents=True, exist_ok=True)
        (output / f"tradeoff_{tag}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        np.savez_compressed(output / f"tradeoff_probs_{tag}.npz", classes=np.asarray(classes),
                            y_val=y_val, y_test=y_test, p1_val=p1_val, p2_val=p2_val,
                            p1_test=p1_test, p2_test=p2_test)
        save_figure(result, PROJECT_ROOT / f"docs/figures/cascade/tradeoff_budget_{tag}.png")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"교환 측정 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
