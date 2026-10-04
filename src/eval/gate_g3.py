"""고정 split에서 교차 적합 라우터와 기존 게이트를 CPU로 확인한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import gate_g2, cascade_tradeoff
from src.models import data_text, gate_signals as gates

SPECIAL = set("% '\"<>();=&|$`{}".replace(" ", ""))


def text_features(texts):
    """문자 비율과 바이트 절삭 신호를 구분하여 다국어 길이를 보존한다."""
    rows = []
    for text in texts:
        length = len(text.encode("utf-8"))
        denominator = max(len(text), 1)
        rows.append([np.log1p(length), sum(c in SPECIAL for c in text) / denominator,
                     text.count("%") / denominator, sum(c.isdigit() for c in text) / denominator,
                     float(length > 2304)])
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def router_features(p1, text_feats, emb=None):
    """표준화 전에 확률·원문·선택적 내부 표현의 열 순서를 고정한다."""
    p1 = gates._probabilities(p1)
    text_feats = np.asarray(text_feats, dtype=float)
    if text_feats.shape != (len(p1), 5) or not np.isfinite(text_feats).all():
        raise ValueError("원문 특징은 유한한 N×5 배열이어야 합니다.")
    top = np.sort(p1, axis=1)[:, -2:]
    parts = [p1, p1.max(1)[:, None], (top[:, 1] - top[:, 0])[:, None], text_feats]
    if emb is not None:
        emb = np.asarray(emb, dtype=float)
        if emb.ndim != 2 or emb.shape[0] != len(p1) or not emb.shape[1] or not np.isfinite(emb).all():
            raise ValueError("임베딩은 행 수가 일치하는 유한한 N×D 배열이어야 합니다.")
        parts.append(emb)
    return np.concatenate(parts, axis=1)


def cross_fit_router(x_val, y_val, target, x_test, seed=42):
    """자신을 학습한 모델의 점수로 val 임계값을 고르는 누출을 막는다."""
    if len(np.unique(target)) != 2:
        raise ValueError("라우터 표적에 두 값이 모두 필요합니다.")
    if np.min(np.unique(y_val, return_counts=True)[1]) < 2:
        raise ValueError("라벨 층화 반분에는 클래스마다 최소 2행이 필요합니다.")
    val_scores = np.empty(len(y_val))
    models, splits = [], []
    for train, held_out in StratifiedKFold(2, shuffle=True, random_state=seed).split(x_val, y_val):
        if len(np.unique(target[train])) != 2:
            raise ValueError("각 반쪽 학습 표적에 두 값이 모두 필요합니다.")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
        model.fit(x_val[train], target[train])
        val_scores[held_out] = model.predict_proba(x_val[held_out])[:, 1]
        models.append(model)
        splits.append((train, held_out))
    test_scores = np.mean([model.predict_proba(x_test)[:, 1] for model in models], axis=0)
    return val_scores, test_scores, models, splits


def folded_weights(model):
    """요청마다 표준화 배열을 만들지 않도록 스케일과 중심을 계수에 접는다."""
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    weight = classifier.coef_[0] / scaler.scale_
    bias = float(classifier.intercept_[0] - np.dot(weight, scaler.mean_))
    return weight, bias


def measure_router_latency(texts, p1, models, requests, seed, emb=None):
    """이미 있는 1차 확률을 제외하고 한 요청의 특징·두 내적·시그모이드를 잰다."""
    if requests <= 0:
        raise ValueError("지연 측정 요청 수는 양수여야 합니다.")
    weights = [folded_weights(model) for model in models]
    features = router_features(p1, text_features(texts), emb)
    for model, (weight, bias) in zip(models, weights):
        if not np.allclose(expit(features @ weight + bias), model.predict_proba(features)[:, 1], atol=1e-9, rtol=0):
            raise ValueError("접어 넣은 계수와 sklearn 점수가 일치하지 않습니다.")
    durations = []
    indices = np.random.default_rng(seed).choice(len(texts), requests, replace=requests > len(texts))
    for index in indices:
        start = time.perf_counter_ns()
        # 검증용 배치 API 대신 단일 행을 만들어 실제 요청의 고정 비용도 포함한다.
        stats = text_features([texts[index]])[0]
        prob = p1[index]
        top = np.sort(prob)[-2:]
        parts = [prob, np.array([prob.max(), top[1] - top[0]]), stats]
        if emb is not None:
            parts.append(emb[index])
        feature = np.concatenate(parts)
        score = sum(float(expit(np.dot(feature, weight) + bias)) for weight, bias in weights) / 2
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(durations))


def aligned_texts(track, split, y, classes):
    """라벨 순서가 하나라도 다르면 확률과 원문 결합을 중단한다."""
    texts, labels = data_text.load_text_split(track, split, "raw")
    encoded = data_text.encode_labels_with(labels, classes)
    if len(texts) != len(y) or not np.array_equal(encoded, y):
        raise ValueError(f"{split} CSV↔npz 라벨 정렬 불일치")
    return texts


def router_curve(p1_val, p2_val, y_val, p1_test, p2_test, y_test, score_val, score_test, step):
    """라우터 점수의 방향만 뒤집어 기존 미만 임계값 규칙을 그대로 사용한다."""
    rows = []
    for budget in np.r_[np.arange(0, 1, step), 1.]:
        tau = gates.tau_for_budget(1 - score_val, budget)
        row = {"budget": float(budget), "tau": tau}
        for split, p1, p2, y, score in (("val", p1_val, p2_val, y_val, score_val),
                                       ("test", p1_test, p2_test, y_test, score_test)):
            pred, _, mask = gates.cascade_apply_signal(p1, p2, 1 - score, tau)
            row.update({f"{split}_escalation_rate": float(mask.mean()),
                        f"{split}_macro_f1": gates._macro_f1_fast(y, pred, p1.shape[1])})
        rows.append(row)
    return rows


def operating_summary(chosen, mask, y, normal_index, ms1, ms2, router_ms=0.):
    """MSP로 마스크를 재계산하지 않고 게이트가 실제 넘긴 행을 집계한다."""
    normal = y == normal_index
    if not normal.any() or normal.all():
        raise ValueError("운영점 재가중에는 정상·공격 행이 모두 필요합니다.")
    rn, ra, pi = float(mask[normal].mean()), float(mask[~normal].mean()), float((~normal).mean())
    prevalences = (("test_actual", pi), ("pi_0.033", .033), ("pi_0.001", .001))
    return {**chosen, "r_N": rn, "r_A": ra, "test_attack_prevalence": pi,
            "deployment": {key: gate_g2.deployment_cost(rn, ra, value, ms1, ms2) for key, value in prevalences},
            "deployment_with_router": {key: gate_g2.deployment_cost(rn, ra, value, ms1 + router_ms, ms2)
                                       for key, value in prevalences}}


def bootstrap_comparison(y, base_mask, base_pred, mask, pred, n_classes, n_boot=2000, seed=42):
    """고정 운영점에 같은 행 인덱스를 적용하여 짝지어진 차이의 CI를 구한다."""
    if n_boot <= 0:
        raise ValueError("부트스트랩 횟수는 양수여야 합니다.")
    rng = np.random.default_rng(seed)
    rate_diffs, f1_diffs = [], []
    for _ in range(n_boot):
        indices = rng.integers(0, len(y), len(y))
        rate_diffs.append(float(mask[indices].mean() - base_mask[indices].mean()))
        f1_diffs.append(gates._macro_f1_fast(y[indices], pred[indices], n_classes)
                        - gates._macro_f1_fast(y[indices], base_pred[indices], n_classes))
    return {"escalation_difference_ci": np.percentile(rate_diffs, [2.5, 97.5]).tolist(),
            "macro_f1_difference_ci": np.percentile(f1_diffs, [2.5, 97.5]).tolist(),
            "n_boot": n_boot, "seed": seed}


def analyze(args):
    """고정 split을 한 fold로 감싸 기존 B0·G2 구현과 같은 경로를 탄다."""
    if not np.isfinite(args.step) or not 0 < args.step <= 1:
        raise ValueError("step은 (0,1]이어야 합니다.")
    gate_g2.deployment_cost(0., 0., 0., args.ms1, args.ms2)
    if args.n_boot <= 0 or args.latency_requests <= 0:
        raise ValueError("부트스트랩·지연 요청 수는 양수여야 합니다.")
    with np.load(args.probs, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("classes", "y_val", "y_test", "p1_val", "p2_val", "p1_test", "p2_test")}
    classes = arrays["classes"].tolist()
    if len(set(classes)) != len(classes) or "Normal" not in classes:
        raise ValueError("classes는 중복이 없고 Normal을 포함해야 합니다.")
    texts = {}
    for split in ("val", "test"):
        p1, p2 = [gates._probabilities(arrays[f"p{i}_{split}"]) for i in (1, 2)]
        if p1.shape != p2.shape or p1.shape[1] != len(classes):
            raise ValueError("확률 모양과 classes가 일치해야 합니다.")
        gates._labels(arrays[f"y_{split}"], len(p1), len(classes))
        texts[split] = aligned_texts(args.track, split, arrays[f"y_{split}"], classes)
    nv, nt = len(arrays["y_val"]), len(arrays["y_test"])
    adapter = {"classes": arrays["classes"], "y": np.r_[arrays["y_val"], arrays["y_test"]],
               "val_idx_1": np.arange(nv), "test_idx_1": nv + np.arange(nt)}
    first = {**adapter, "val_probs_1": arrays["p1_val"], "test_probs_1": arrays["p1_test"]}
    second = {**adapter, "val_probs_1": arrays["p2_val"], "test_probs_1": arrays["p2_test"]}
    fold = gate_g2.fold_curves(first, second, [1], args.step)[0]
    curves = fold["curves"]
    confidences = {"b0": arrays["p1_test"].max(1), "g2": arrays["p1_test"].max(1)}
    latencies = {}
    embeddings = {"g3_lite": (None, None)}
    if args.embeddings:
        with np.load(args.embeddings, allow_pickle=False) as archive:
            embeddings["g3_full"] = (archive["val"], archive["test"])
    target = (arrays["p1_val"].argmax(1) != arrays["y_val"]) & (arrays["p2_val"].argmax(1) == arrays["y_val"])
    for name, (emb_val, emb_test) in embeddings.items():
        xv = router_features(arrays["p1_val"], text_features(texts["val"]), emb_val)
        xt = router_features(arrays["p1_test"], text_features(texts["test"]), emb_test)
        sv, st, models, splits = cross_fit_router(xv, arrays["y_val"], target, xt, args.seed)
        curves[name] = router_curve(*(arrays[key] for key in ("p1_val", "p2_val", "y_val", "p1_test", "p2_test", "y_test")), sv, st, args.step)
        confidences[name] = 1 - st
        latencies[name] = measure_router_latency(texts["test"], arrays["p1_test"], models, args.latency_requests, args.seed, emb_test)
    teacher = fold["stage2_test_macro_f1"]
    offset = 0. if any(row["test_macro_f1"] >= teacher for row in curves["b0"]) else .005
    try:
        strata, strata_reason = cascade_tradeoff.load_strata(args.track, None, arrays["y_test"], classes)
    except FileNotFoundError as error:
        strata, strata_reason = None, str(error)
    summary = {"label_alignment": {"val": nv, "test": nt, "passed": True},
               "stage2_val_macro_f1": fold["stage2_val_macro_f1"], "stage2_test_macro_f1": teacher,
               "used_target_offset": offset, "oracle": fold["oracle"], "gates": {}}
    masks, predictions = {}, {}
    for name, curve in curves.items():
        eligible = [row for row in curve if row["val_macro_f1"] >= fold["stage2_val_macro_f1"] - .005]
        if not eligible:
            raise ValueError(f"{name}: val 목표 미도달")
        chosen = min(eligible, key=lambda row: row["val_escalation_rate"])
        mask = (gates.apply_classwise(arrays["p1_test"], confidences[name], chosen["taus"])
                if name == "g2" else confidences[name] < chosen["tau"])
        masks[name] = mask
        predictions[name] = np.where(mask, arrays["p2_test"].argmax(1), arrays["p1_test"].argmax(1))
        ms = latencies.get(name, 0.)
        op = operating_summary(chosen, mask, arrays["y_test"], classes.index("Normal"), args.ms1, args.ms2, ms)
        reached = [row["test_escalation_rate"] for row in curve if row["test_macro_f1"] >= teacher - offset]
        needed = min(reached) if reached else None
        item = {"operating_point": op, "needed_test_escalation": needed,
                "over_escalation_ratio": needed / fold["oracle"] if needed is not None and fold["oracle"] else None,
                "normal_strata": {"reason": strata_reason}}
        if strata is not None:
            item["normal_strata"] = {}
            for stratum in ("real_user", "scanner", "mixed"):
                selection = (np.asarray(strata) == stratum) & (arrays["y_test"] == classes.index("Normal"))
                item["normal_strata"][stratum] = {"rows": int(selection.sum()),
                                                     "escalation_rate": float(mask[selection].mean()) if selection.any() else None}
        if name in latencies:
            item.update(router_ms=ms, router_cost_ok=ms <= .035,
                        router_cost_note="비용 조건 충족" if ms <= .035 else "라우터 비용 조건 미달")
        summary["gates"][name] = item
    for name, flag in (("g2", "H-G2c"), ("g3_lite", "H-G3L"), ("g3_full", "H-G3F")):
        if name not in curves:
            continue
        comparison = bootstrap_comparison(arrays["y_test"], masks["b0"], predictions["b0"], masks[name], predictions[name], len(classes), args.n_boot, args.seed)
        base, current = [summary["gates"][key]["operating_point"] for key in ("b0", name)]
        drop = (base["test_escalation_rate"] - current["test_escalation_rate"]) * 100
        loss = (base["test_macro_f1"] - current["test_macro_f1"]) * 100
        upper_negative = comparison["escalation_difference_ci"][1] < 0
        summary["gates"][name].update(bootstrap=comparison, hypothesis={"name": flag,
            "escalation_drop_pp": drop, "ci_upper_lt_zero": upper_negative, "f1_loss_pp": loss,
            "adopted": bool(drop >= 5 and upper_negative and loss <= .3)})
    return {"probs": args.probs.name, "embeddings": args.embeddings.name if args.embeddings else None,
            "classes": classes, "step": args.step, "seed": args.seed, "ms1": args.ms1, "ms2": args.ms2,
            "latency_requests": args.latency_requests, "latency_note": "T2 CPU·배치1 지연을 재사용한 근사; 내부 표현 추출 비용 제외",
            "curves": curves, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--track", default="srbh_4class")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--step", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--ms1", type=float, default=.350)
    parser.add_argument("--ms2", type=float, default=1.114)
    parser.add_argument("--latency-requests", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args)
        output = args.out or PROJECT_ROOT / "experiments/results" / f"gate_g3_{args.probs.stem}{'_emb' if args.embeddings else ''}.json"
        content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        # 연구 결과의 조용한 덮어쓰기를 막기 위해 배타적으로 생성한다.
        with output.open("x", encoding="utf-8") as saved:
            saved.write(content)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(f"G3 게이트 비교 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
