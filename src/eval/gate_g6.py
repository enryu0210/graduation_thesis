"""H-G6f(docs/14 §8.14) — 권장 구성의 고정 split 에서 G6 거리 점수를 확인하고 부트스트랩 CI 를 낸다.

고정 split 을 한 fold 로 감싸 CV 판정(gate_g6_cv)과 같은 운영점·지표 경로를 탄다.
CV 에서 기각된 점수의 값은 참고로만 쓴다(논문 "개선"은 H-G6 과 H-G6f 동시 충족일 때만).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cascade_tradeoff, gate_g2
from src.eval.gate_g6_cv import expected_rate, fold_gates
from src.models import feature_signals as fs
from src.models import gate_signals as gates


def bootstrap_e_pi(y, normal_index, base_mask, base_pred, mask, pred, n_classes, n_boot, seed):
    """임계값·마스크를 고정한 채 test 행을 재표본해 E_π·Macro-F1 차이(점수−B0)의 95% CI 를 낸다."""
    rng = np.random.default_rng(seed)
    e_diffs, f1_diffs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        normal = y[idx] == normal_index

        def e_pi(m):
            return expected_rate(m[idx][normal].mean(), m[idx][~normal].mean())

        e_diffs.append(e_pi(mask) - e_pi(base_mask))
        f1_diffs.append(gates._macro_f1_fast(y[idx], pred[idx], n_classes)
                        - gates._macro_f1_fast(y[idx], base_pred[idx], n_classes))
    return {"E_pi_difference_ci": np.percentile(e_diffs, [2.5, 97.5]).tolist(),
            "macro_f1_difference_ci": np.percentile(f1_diffs, [2.5, 97.5]).tolist(),
            "n_boot": n_boot, "seed": seed}


def analyze(args):
    with np.load(args.probs, allow_pickle=False) as archive:
        a = {key: archive[key] for key in ("classes", "y_val", "y_test", "p1_val", "p2_val",
                                           "p1_test", "p2_test")}
    with np.load(args.embeddings, allow_pickle=False) as archive:
        emb = {key: archive[key] for key in archive.files}
    with np.load(args.train_embeddings, allow_pickle=False) as archive:
        emb_tr = {key: archive[key] for key in archive.files}
    classes = a["classes"].tolist()
    # 표현 파일이 같은 행 순서인지: 라벨 배열 전체와 행 수, 클래스 순서를 대조한다.
    for split in ("val", "test"):
        if not np.array_equal(emb[f"y_{split}"], a[f"y_{split}"]) or len(emb[split]) != len(a[f"p1_{split}"]):
            raise ValueError(f"{split} 표현과 확률 파일의 라벨·행 수가 다릅니다.")
    for source in (emb, emb_tr):
        if "classes" in source and source["classes"].tolist() != classes:
            raise ValueError("표현 파일의 클래스 순서가 다릅니다.")
    nv, nt = len(a["y_val"]), len(a["y_test"])
    adapter = {"classes": a["classes"], "y": np.r_[a["y_val"], a["y_test"]],
               "val_idx_1": np.arange(nv), "test_idx_1": nv + np.arange(nt)}
    first = {**adapter, "val_probs_1": a["p1_val"], "test_probs_1": a["p1_test"]}
    second = {**adapter, "val_probs_1": a["p2_val"], "test_probs_1": a["p2_test"]}
    row = gate_g2.fold_curves(first, second, [1], args.step)[0]
    teacher = row["stage2_test_macro_f1"]
    offset = 0. if any(p["test_macro_f1"] >= teacher for p in row["curves"]["b0"]) else .005
    row["target_macro_f1"] = teacher - offset
    strata, _ = cascade_tradeoff.load_strata("srbh_4class", None, a["y_test"], classes)
    strata = np.array([s if isinstance(s, str) else "" for s in strata])  # 공격 행은 NaN → ""
    result, diag = fold_gates(row, a["p1_val"], a["p2_val"], a["y_val"], a["p1_test"], a["p2_test"],
                              a["y_test"], emb_tr["train"], emb_tr["y_train"], emb["val"], emb["test"],
                              classes, args.step, args.latency_requests, args.seed, strata)
    normal = classes.index("Normal")
    first_pred, second_pred = a["p1_test"].argmax(1), a["p2_test"].argmax(1)
    preds = {g: np.where(item["test_mask"], second_pred, first_pred)
             for g, item in result.items() if item["operating_point"]}
    base = result["b0"]["operating_point"]
    verdicts = {}
    for g in ("g2", *fs.SCORE_NAMES, *(f"{n}_cw" for n in fs.SCORE_NAMES)):
        op = result[g]["operating_point"]
        if op is None:
            verdicts[g] = {"adopted": False, "reason": "val 목표 미도달"}
            continue
        boot = bootstrap_e_pi(a["y_test"], normal, result["b0"]["test_mask"], preds["b0"],
                              result[g]["test_mask"], preds[g], len(classes), args.n_boot, args.seed)
        reduction = 1 - op["E_pi"] / base["E_pi"]
        loss_pp = (base["test_macro_f1"] - op["test_macro_f1"]) * 100
        verdicts[g] = {"reduction": reduction, "f1_loss_pp": loss_pp, **boot,
                       "adopted": bool(boot["E_pi_difference_ci"][1] < 0 and reduction >= .20 and loss_pp <= .3),
                       "exploratory": g == "g2" or g.endswith("_cw")}
    for item in result.values():
        item.pop("test_mask", None)
    return {"probs": args.probs.name, "embeddings": args.embeddings.name,
            "train_embeddings": args.train_embeddings.name, "classes": classes,
            "n_train": int(len(emb_tr["train"])), "n_val": nv, "n_test": nt,
            "stage2_val_macro_f1": row["stage2_val_macro_f1"], "stage2_test_macro_f1": teacher,
            "used_offset": offset, "oracle": row["oracle"],
            "gates": result, "diagnostics": diag, "H-G6f": verdicts,
            "note": "g2·*_cw 는 판정 미사용(참고/탐색). r_N_strata 는 test 실사용자 행이 적어 참고값."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probs", "embeddings", "train_embeddings"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--step", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--latency-requests", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        output = args.out or PROJECT_ROOT / f"experiments/results/gate_g6_{args.probs.stem}.json"
        if output.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output.name}")
        result = analyze(args)
        with output.open("x", encoding="utf-8") as saved:
            saved.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        brief = {g: {k: v for k, v in item["operating_point"].items()
                     if k in ("E_pi", "r_N", "r_A", "test_escalation_rate", "test_macro_f1", "r_N_strata")}
                 for g, item in result["gates"].items() if item["operating_point"]}
        print(json.dumps({"operating": brief, "H-G6f": result["H-G6f"],
                          "diagnostics": result["diagnostics"]}, ensure_ascii=False, indent=2))
        print(f"저장: {output.name}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"G6 고정 split 확인 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
