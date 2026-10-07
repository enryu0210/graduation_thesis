"""G6 · G4(docs/14 §8.14) — 1차 표현 거리 점수로 넘김을 정할 때 배포 넘김률이 줄어드는지 CV 로 판정한다.

같은 fold 에서 B0(msp 단일 τ)·G2(msp 클래스별 τ)·거리 점수 4개(단일 τ, 탐색용 클래스별 τ)를 비교한다.
판정 지표는 test 비율 넘김률이 아니라 **배포 재가중 넘김률 E_π**(π=3.3%)다 — 정상 넘김(r_N)을 늘린
게이트가 test 넘김률만 보고 통과했던 §8.9 의 오판을 막기 위해서다.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cross_validate, gate_cv, gate_g2, gate_g3, gate_g3_cv
from src.eval.cv_compare import holm_bonferroni
from src.models import feature_signals as fs
from src.models import gate_signals as gates

PI = .033
MS1, MS2 = .350, 1.114        # T2 CPU·배치1 실측(1차 RGB CNN / 2차 char-CNN)
COST_LIMIT_MS = .035          # 1차 지연의 10% (§8.14 ④)
STRATA = ("real_user", "scanner", "mixed")


def expected_rate(r_normal, r_attack, prevalence=PI):
    return prevalence * r_attack + (1 - prevalence) * r_normal


def speedup(rate, score_ms):
    return MS2 / (MS1 + score_ms + rate * MS2)


def gate_mask(point, p1, conf):
    """클래스별 운영점은 1차 예측 클래스의 τ 를, 단일 운영점은 τ 하나를 쓴다."""
    return (gates.apply_classwise(p1, conf, point["taus"]) if "taus" in point
            else conf < point["tau"])


def signal_curves(p1v, p2v, yv, p1t, p2t, yt, conf_val, conf_test, step):
    """점수를 '클수록 확신' 확신도로 받아 단일 τ 곡선과 클래스별 τ 경로를 만든다."""
    single = gate_g3.router_curve(p1v, p2v, yv, p1t, p2t, yt, 1 - conf_val, 1 - conf_test, step)
    classwise = gates.classwise_tau_path(p1v, p2v, yv, p1v.shape[1], step, conf_val=conf_val)
    for point in classwise:
        mask = gates.apply_classwise(p1t, conf_test, point["taus"])
        pred = np.where(mask, p2t.argmax(1), p1t.argmax(1))
        point.update(test_escalation_rate=float(mask.mean()),
                     test_macro_f1=gates._macro_f1_fast(yt, pred, p1t.shape[1]))
    return single, classwise


def evaluate_gate(curve, conf_val, conf_test, ctx, score_ms=0.):
    """val 규칙으로 운영점을 고르고 test 에서 r_N·r_A·E_π·층별 r_N 을 잰다."""
    reached = [row["test_escalation_rate"] for row in curve if row["test_macro_f1"] >= ctx["target"]]
    needed = min(reached) if reached else None
    out = {"needed_test_escalation": needed,
           "ratio": needed / ctx["oracle"] if needed is not None and ctx["oracle"] else None,
           "score_ms": score_ms, "operating_point": None}
    eligible = [row for row in curve if row["val_macro_f1"] >= ctx["s2_val"] - .005]
    if not eligible:
        return out
    chosen = min(eligible, key=lambda row: row["val_escalation_rate"])
    mask_v, mask_t = gate_mask(chosen, ctx["p1v"], conf_val), gate_mask(chosen, ctx["p1t"], conf_test)
    normal_v, normal_t = ctx["yv"] == ctx["normal"], ctx["yt"] == ctx["normal"]
    r_n, r_a = float(mask_t[normal_t].mean()), float(mask_t[~normal_t].mean())
    pi_test = float((~normal_t).mean())
    op = {key: chosen[key] for key in chosen if key not in ("allocations",)}
    op.update(r_N=r_n, r_A=r_a, E_pi=expected_rate(r_n, r_a),
              r_N_val=float(mask_v[normal_v].mean()), test_attack_prevalence=pi_test,
              speedup={name: speedup(expected_rate(r_n, r_a, pi), score_ms)
                       for name, pi in (("test_actual", pi_test), ("pi_0.033", PI), ("pi_0.001", .001))})
    if ctx.get("strata_test") is not None:
        op["r_N_strata"] = {s: (float(mask_t[ctx["strata_test"] == s].mean())
                                if (ctx["strata_test"] == s).any() else None) for s in STRATA}
        real = op["r_N_strata"]["real_user"]
        if real is not None:
            op["g4_gap"] = real - op["r_N_val"]
            op["speedup_real_user_pi_0.033"] = speedup(expected_rate(real, r_a), score_ms)
    out.update(operating_point=op, test_mask=mask_t)
    return out


def fold_gates(row, p1v, p2v, yv, p1t, p2t, yt, emb_tr, y_tr, emb_v, emb_t, classes, step,
               latency_requests, seed, strata_test=None):
    """한 fold(또는 고정 split)의 모든 게이트. row 는 gate_g2.fold_curves 의 결과(b0·g2 곡선 포함)."""
    ref = fs.fit_reference(emb_tr, y_tr, len(classes), seed=seed)
    score_v, score_t = fs.scores(ref, emb_v, p1v), fs.scores(ref, emb_t, p1t)
    latency = fs.measure_latency(ref, emb_t, p1t, latency_requests, seed)
    ctx = {"p1v": p1v, "p1t": p1t, "yv": yv, "yt": yt, "normal": classes.index("Normal"),
           "s2_val": row["stage2_val_macro_f1"], "target": row["target_macro_f1"],
           "oracle": row["oracle"], "strata_test": strata_test}
    msp_v, msp_t = p1v.max(1), p1t.max(1)
    gates_out = {"b0": evaluate_gate(row["curves"]["b0"], msp_v, msp_t, ctx),
                 "g2": evaluate_gate(row["curves"]["g2"], msp_v, msp_t, ctx)}
    wrong = p1t.argmax(1) != yt
    diagnostics = {"msp": {"misclass_auroc": float(roc_auc_score(wrong, -msp_t))}}
    for name in fs.SCORE_NAMES:
        # router_curve 는 1−점수를 확신도로 쓰므로 같은 변환으로 맞춘다(순위만 쓰이므로 정보 손실 없음).
        cv_, ct_ = 1 - score_v[name], 1 - score_t[name]
        single, classwise = signal_curves(p1v, p2v, yv, p1t, p2t, yt, cv_, ct_, step)
        gates_out[name] = evaluate_gate(single, cv_, ct_, ctx, latency[name])
        gates_out[f"{name}_cw"] = evaluate_gate(classwise, cv_, ct_, ctx, latency[name])
        diagnostics[name] = {"misclass_auroc": float(roc_auc_score(wrong, score_t[name])),
                             "spearman_with_msp": float(spearmanr(score_t[name], -msp_t).statistic),
                             "score_ms": latency[name]}
    return gates_out, diagnostics


def load_train_embeddings(path, first, folds):
    """train 표현이 같은 풀·같은 라벨을 가리키고 그 fold 의 val/test 와 겹치지 않는지 확인한다."""
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    expected = {"classes", "y", "label_fingerprint"} | {
        f"train_{kind}_{k}" for k in folds for kind in ("idx", "emb")}
    if set(data) != expected:
        raise ValueError("train 표현 파일의 키가 일치하지 않습니다.")
    for key in ("classes", "y", "label_fingerprint"):
        if not np.array_equal(data[key], first[key]):
            raise ValueError(f"train 표현 파일의 {key}가 다릅니다.")
    for k in folds:
        idx = data[f"train_idx_{k}"]
        held = np.concatenate([first[f"val_idx_{k}"], first[f"test_idx_{k}"]])
        if np.intersect1d(idx, held).size or len(idx) + len(held) != len(first["y"]):
            raise ValueError(f"fold {k} train 인덱스가 val/test 와 겹치거나 풀을 덮지 않습니다.")
        if data[f"train_emb_{k}"].shape != (len(idx), 128):
            raise ValueError(f"fold {k} train 표현 모양이 다릅니다.")
    return data


def _mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def summarize(fold_rows, gate_names):
    """fold 평균과 H-G6(4쌍 Holm)·H-G4 판정. 판정 수치는 §8.14 그대로."""
    ops = {g: [row["gates"][g]["operating_point"] for row in fold_rows] for g in gate_names}
    table = {}
    for g in gate_names:
        if any(op is None for op in ops[g]):
            table[g] = {"reason": "val 목표 미도달 fold 가 있습니다."}
            continue
        table[g] = {f"mean_{key}": _mean([op[key] for op in ops[g]])
                    for key in ("E_pi", "r_N", "r_A", "test_escalation_rate", "test_macro_f1", "r_N_val")}
        table[g]["mean_ratio"] = _mean([row["gates"][g]["ratio"] for row in fold_rows])
        table[g]["mean_score_ms"] = _mean([row["gates"][g]["score_ms"] for row in fold_rows])
        table[g]["mean_speedup"] = {key: _mean([op["speedup"][key] for op in ops[g]])
                                    for key in ("test_actual", "pi_0.033", "pi_0.001")}
        table[g]["per_fold_E_pi"] = [op["E_pi"] for op in ops[g]]
        if all("g4_gap" in op for op in ops[g]):
            table[g]["r_N_strata"] = {s: _mean([op["r_N_strata"][s] for op in ops[g]]) for s in STRATA}
            table[g]["gaps"] = [op["g4_gap"] for op in ops[g]]
            table[g]["mean_gap"] = _mean(table[g]["gaps"])
            table[g]["mean_speedup_real_user_pi_0.033"] = _mean(
                [op["speedup_real_user_pi_0.033"] for op in ops[g]])
    base = table["b0"]["per_fold_E_pi"]
    raw = {}
    for name in fs.SCORE_NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw[name] = float(ttest_rel(table[name]["per_fold_E_pi"], base).pvalue) \
                if "per_fold_E_pi" in table[name] else float("nan")
    finite = {n: p for n, p in raw.items() if np.isfinite(p)}
    holm = dict(zip(finite, holm_bonferroni(list(finite.values())))) if finite else {}
    verdicts = {}
    for name in fs.SCORE_NAMES:
        item = table[name]
        if "mean_E_pi" not in item:
            verdicts[name] = {"adopted": False, "reason": item["reason"]}
            continue
        reduction = 1 - item["mean_E_pi"] / table["b0"]["mean_E_pi"]
        loss_pp = (table["b0"]["mean_test_macro_f1"] - item["mean_test_macro_f1"]) * 100
        c = {"c1_reduction_ge_20pct": reduction >= .20,
             # 양측 검정이라 '더 나빠짐'도 유의하게 나온다 → 개선 방향일 때만 충족으로 본다.
             "c2_holm_p_lt_0.05": holm.get(name, 1.) < .05 and reduction > 0,
             "c3_f1_loss_le_0.3pp": loss_pp <= .3,
             "c4_score_ms_le_0.035": item["mean_score_ms"] <= COST_LIMIT_MS}
        verdicts[name] = {"reduction": reduction, "raw_p": raw[name], "holm_p": holm.get(name),
                          "f1_loss_pp": loss_pp, "score_ms": item["mean_score_ms"], **c,
                          "adopted": all(c.values()),
                          "note": ("넘김은 줄지만 비용 조건 미달"
                                   if all(list(c.values())[:3]) and not c["c4_score_ms_le_0.035"] else None)}
    adopted = [n for n in fs.SCORE_NAMES if verdicts[n]["adopted"]]
    candidates = [n for n in fs.SCORE_NAMES if "mean_E_pi" in table[n]]
    designated = adopted[0] if adopted else min(candidates, key=lambda n: table[n]["mean_E_pi"])
    g4 = {"designated_signal": designated, "designated_reason": "채택" if adopted else "E_π 최저(참고)"}
    gaps = table["b0"].get("gaps")
    if gaps is not None:
        p = float(ttest_1samp(gaps, 0.).pvalue)
        g4.update(b0_mean_gap=float(np.mean(gaps)), b0_p=p,
                  **{"H-G4": bool(np.mean(gaps) >= .10 and p < .05)})
    return {"gates": table, "H-G6": verdicts, "adopted": adopted, "G4": g4}


def analyze(args):
    if not np.isfinite(args.step) or not 0 < args.step <= 1:
        raise ValueError("step은 (0,1]이어야 합니다.")
    first, second, folds = gate_cv.validate_alignment(args.stage1, args.stage2)
    emb = gate_g3_cv.load_embeddings(args.embeddings, first, folds)
    train_emb = load_train_embeddings(args.train_embeddings, first, folds)
    classes = first["classes"].tolist()
    strata = cross_validate.load_pool_strata("srbh_4class", exclude_test=True)
    if len(strata) != len(first["y"]):
        raise ValueError("층 배열과 CV 풀의 행 수가 다릅니다.")
    rows = gate_g2.fold_curves(first, second, folds, args.step)
    failed = [r["fold"] for r in rows if not any(
        p["test_macro_f1"] >= r["stage2_test_macro_f1"] for p in r["curves"]["b0"])]
    offset = .005 if failed else 0.     # gate_g3_cv 와 같은 과잉 넘김 배율 목표 규칙
    fold_rows = []
    for row in rows:
        k = row["fold"]
        vi, ti, tri = first[f"val_idx_{k}"], first[f"test_idx_{k}"], train_emb[f"train_idx_{k}"]
        row["target_macro_f1"] = row["stage2_test_macro_f1"] - offset
        result, diag = fold_gates(
            row, first[f"val_probs_{k}"], second[f"val_probs_{k}"], first["y"][vi],
            first[f"test_probs_{k}"], second[f"test_probs_{k}"], first["y"][ti],
            train_emb[f"train_emb_{k}"], first["y"][tri], emb[f"val_emb_{k}"], emb[f"test_emb_{k}"],
            classes, args.step, args.latency_requests, args.seed, strata[ti])
        for item in result.values():
            item.pop("test_mask", None)
        fold_rows.append({"fold": k, "oracle": row["oracle"], "n_train": int(len(tri)),
                          "stage2_test_macro_f1": row["stage2_test_macro_f1"],
                          "gates": result, "diagnostics": diag})
        print(f"fold {k}: " + " ".join(
            f"{g}={result[g]['operating_point']['E_pi']:.4f}" for g in ("b0", "g2", *fs.SCORE_NAMES)
            if result[g]["operating_point"]), flush=True)
    names = ["b0", "g2", *fs.SCORE_NAMES, *(f"{n}_cw" for n in fs.SCORE_NAMES)]
    summary = summarize(fold_rows, names)
    summary["diagnostics"] = {name: {key: _mean([r["diagnostics"][name][key] for r in fold_rows])
                                     for key in fold_rows[0]["diagnostics"][name]}
                              for name in fold_rows[0]["diagnostics"]}
    summary["b0_ratio_per_fold"] = [r["gates"]["b0"]["ratio"] for r in fold_rows]
    return {"stage1": args.stage1.name, "stage2": args.stage2.name,
            "embeddings": args.embeddings.name, "train_embeddings": args.train_embeddings.name,
            "classes": classes, "label_fingerprint": first["label_fingerprint"].item(),
            "step": args.step, "seed": args.seed, "pi": PI, "ms1": MS1, "ms2": MS2,
            "used_offset": offset, "initial_b0_unreached_folds": failed,
            "note": "*_cw 는 탐색(판정 미사용). 점수 지연은 CPU 단일 요청 중앙값, 1차 표현 추출 비용 제외.",
            "folds": fold_rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1", "stage2", "embeddings", "train_embeddings"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--step", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-requests", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        output = args.out or PROJECT_ROOT / f"experiments/results/gate_g6_cv_{args.stage1.stem}.json"
        if output.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output.name}")
        result = analyze(args)
        with output.open("x", encoding="utf-8") as saved:
            saved.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        summary = result["summary"]
        print(json.dumps({key: summary[key] for key in ("H-G6", "adopted", "G4", "diagnostics",
                                                        "b0_ratio_per_fold")},
                         ensure_ascii=False, indent=2))
        print(f"저장: {output.name}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"G6 CV 판정 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
