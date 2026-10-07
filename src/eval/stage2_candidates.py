"""T3(docs/14 §8.14) — 권장 1차 위에 2차 후보 세 개를 조립해 (배포 기대 지연, test Macro-F1) 곡선을 비교한다.

2차 후보: char-CNN `_bal`(기준) · BiLSTM `--balance` · TF-IDF+LogReg(class_weight balanced, 여기서 학습).
넘김은 1차 msp 단일 τ(B0)라 **넘김 마스크는 후보와 무관하게 같다** — 곡선을 가르는 것은 2차의 정확도와 지연 t2 뿐이다.
1차 확률은 T1 `_s2bal` 확률 파일에서 읽고(재추론 없음), 지연은 이 실행에서 CPU·배치1 로 다시 잰다.
서술 판정: 후보의 어떤 점이 같거나 작은 지연의 char-CNN 최고 F1 보다 높고 부트스트랩 CI 하한 > 0 이면 "대안 2차로 병기".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval.cascade_tradeoff import select_operating_point
from src.eval.gate_g6_cv import PI, expected_rate
from src.models import cascade
from src.models import gate_signals as gates

STEP = .005
CANDIDATES = ("charcnn", "bilstm", "tfidf")


def fit_tfidf(train_texts, y_train, n_classes, max_features=20000):
    """baseline_tfidf.py 와 같은 설정(문자 2~4gram, LogReg balanced)이다 — 설정을 바꾸면 T3 비교가 아니게 된다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 max_features=max_features, lowercase=False)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced", C=1.0)
    clf.fit(vectorizer.fit_transform(train_texts), y_train)
    if list(clf.classes_) != list(range(n_classes)):
        raise ValueError("TF-IDF 분류기의 클래스 순서가 트랙과 다릅니다.")
    return vectorizer, clf


def tfidf_latency_ms(vectorizer, clf, texts, requests, seed):
    """요청 1건의 변환+예측 시간 중앙값. 전처리를 포함하는 것은 TF-IDF 는 변환이 곧 모델 입력 계산이라서다(§8.14)."""
    rows = np.random.default_rng(seed).integers(0, len(texts), requests)
    timings = []
    for row in rows:
        started = time.perf_counter_ns()
        clf.predict_proba(vectorizer.transform([texts[row]]))
        timings.append(time.perf_counter_ns() - started)
    return float(np.median(timings) / 1e6)


def torch_latency_ms(net, x, seed):
    """cascade_tradeoff.measure_devices 와 같은 CPU·배치1 측정(무작위 표본 1개, 라운드 중앙값)."""
    cpu = torch.device("cpu")
    net.to(cpu)
    index = np.random.default_rng(seed).choice(len(x), 1)
    return cascade.measure_latency(net, x, cpu, 1, sample_idx=index)


def budget_masks(p1_val, p1_test, step):
    """예산 격자마다 val msp 로 τ 를 정하고 val·test 넘김 마스크를 만든다(후보 공통)."""
    val_msp, test_msp = p1_val.max(1), p1_test.max(1)
    out = []
    for budget in np.round(np.arange(0, 1 + step / 2, step), 6):
        tau = gates.tau_for_budget(val_msp, budget)
        out.append((float(budget), tau, val_msp < tau, test_msp < tau))
    return out


def candidate_curve(masks, p1_val, p1_test, p2_val, p2_test, y_val, y_test, normal, t1, t2):
    n = p1_val.shape[1]
    attack = y_test != normal
    rows = []
    for budget, tau, mv, mt in masks:
        pred_v = np.where(mv, p2_val.argmax(1), p1_val.argmax(1))
        pred_t = np.where(mt, p2_test.argmax(1), p1_test.argmax(1))
        e_pi = expected_rate(mt[~attack].mean(), mt[attack].mean())
        rows.append({"budget": budget, "tau": tau, "val_macro_f1": gates._macro_f1_fast(y_val, pred_v, n),
                     "test_macro_f1": gates._macro_f1_fast(y_test, pred_t, n),
                     "test_escalation_rate": float(mt.mean()), "E_pi": e_pi,
                     "latency_ms_pi": t1 + e_pi * t2, "speedup_pi": t2 / (t1 + e_pi * t2)})
    return rows


def bootstrap_f1_diff(y, pred_a, pred_b, n_classes, n_boot, seed):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(gates._macro_f1_fast(y[idx], pred_a[idx], n_classes)
                     - gates._macro_f1_fast(y[idx], pred_b[idx], n_classes))
    return np.percentile(diffs, [2.5, 97.5]).tolist()


def dominance(curves, masks, preds2, p1_test, y_test, n_classes, n_boot, seed):
    """후보 점마다 '지연이 같거나 작은 char-CNN 점 중 최고 F1'과 비교한다(그 점의 CI 로 판정)."""
    base = curves["charcnn"]
    first = p1_test.argmax(1)
    result = {}
    for name in CANDIDATES[1:]:
        best = None
        for i, row in enumerate(curves[name]):
            ref = [j for j, b in enumerate(base) if b["latency_ms_pi"] <= row["latency_ms_pi"]]
            if not ref:
                continue
            j = max(ref, key=lambda k: base[k]["test_macro_f1"])
            gain = row["test_macro_f1"] - base[j]["test_macro_f1"]
            if gain > 0 and (best is None or gain > best["gain"]):
                best = {"gain": gain, "candidate": row, "charcnn_ref": base[j], "i": i, "j": j}
        if best is None:
            result[name] = {"listed_as_alternative": False, "reason": "같은 지연 이하의 char-CNN 점보다 높은 F1 점이 없다"}
            continue
        pred_c = np.where(masks[best["i"]][3], preds2[name], first)
        pred_b = np.where(masks[best["j"]][3], preds2["charcnn"], first)
        ci = bootstrap_f1_diff(y_test, pred_c, pred_b, n_classes, n_boot, seed)
        result[name] = {"best_gain_pp": best["gain"] * 100, "f1_diff_ci": ci,
                        "candidate_point": best["candidate"], "charcnn_point": best["charcnn_ref"],
                        "listed_as_alternative": bool(ci[0] > 0)}
    return result


def analyze(args):
    import data_text
    with np.load(args.probs, allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    classes = a["classes"].tolist()
    normal, n = classes.index("Normal"), len(classes)
    texts = {s: data_text.load_text_split("srbh_4class", s, "raw") for s in ("train", "val", "test")}
    y = {s: np.asarray(data_text.encode_labels_with(texts[s][1], classes)) for s in texts}
    for s in ("val", "test"):
        if not np.array_equal(y[s], a[f"y_{s}"]):
            raise ValueError(f"{s} CSV 라벨이 확률 파일과 다릅니다(행 순서 불일치).")
    if args.limit:   # 스모크: 행 수만 줄인다(저장하지 않는다)
        for s in ("val", "test"):
            a[f"p1_{s}"], a[f"p2_{s}"], y[s] = a[f"p1_{s}"][:args.limit], a[f"p2_{s}"][:args.limit], y[s][:args.limit]
            texts[s] = (texts[s][0][:args.limit], texts[s][1][:args.limit])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_seq = {s: torch.from_numpy(data_text.encode_byte_matrix(texts[s][0], 2304)) for s in ("val", "test")}
    probs2 = {"charcnn": (a["p2_val"], a["p2_test"])}
    nets = {"charcnn": cascade.load_net("charcnn", n, "srbh_4class", "raw", True, device)}
    nets["bilstm"] = cascade.load_net("bilstm", n, "srbh_4class", "raw", True, device)
    probs2["bilstm"] = tuple(cascade.predict_probs(nets["bilstm"], x_seq[s], device, 256) for s in ("val", "test"))
    train_texts = texts["train"][0][:args.limit * 10] if args.limit else texts["train"][0]
    vectorizer, clf = fit_tfidf(train_texts, y["train"][:len(train_texts)], n)
    probs2["tfidf"] = tuple(clf.predict_proba(vectorizer.transform(texts[s][0])) for s in ("val", "test"))
    # 지연: 1차와 2차 후보를 같은 실행·같은 CPU 조건에서 잰다(§8.11 수치와 섞지 않는다).
    net1 = cascade.load_net("cnn", n, "srbh_4class", "raw", False, device, "rgb", 3,
                            ("raw_byte", "char_class", "local_entropy"))
    x_img = cascade.build_inputs("srbh_4class", "test", "raw", 48, "rgb",
                                 ("raw_byte", "char_class", "local_entropy"), 2304, 2000)[0]
    t1 = torch_latency_ms(net1, x_img, args.seed)
    t2 = {name: torch_latency_ms(nets[name], x_seq["test"], args.seed) for name in ("charcnn", "bilstm")}
    t2["tfidf"] = tfidf_latency_ms(vectorizer, clf, texts["test"][0], args.latency_requests, args.seed)
    masks = budget_masks(a["p1_val"], a["p1_test"], STEP)
    curves = {name: candidate_curve(masks, a["p1_val"], a["p1_test"], *probs2[name], y["val"], y["test"],
                                    normal, t1, t2[name]) for name in CANDIDATES}
    alone = {name: {"val_macro_f1": gates._macro_f1_fast(y["val"], probs2[name][0].argmax(1), n),
                    "test_macro_f1": gates._macro_f1_fast(y["test"], probs2[name][1].argmax(1), n)}
             for name in CANDIDATES}
    operating = {name: select_operating_point(curves[name], alone[name]["val_macro_f1"]) for name in CANDIDATES}
    preds2 = {name: probs2[name][1].argmax(1) for name in CANDIDATES}
    verdict = dominance(curves, masks, preds2, a["p1_test"], y["test"], n, args.n_boot, args.seed)
    return {"probs": args.probs.name, "classes": classes, "pi": PI, "step": STEP,
            "t1_ms_cpu1": t1, "t2_ms_cpu1": t2, "stage1_test_macro_f1":
                gates._macro_f1_fast(y["test"], a["p1_test"].argmax(1), n),
            "stage2_alone": alone, "operating_point": operating, "H-T3": verdict, "curves": curves}


def save_figure(result, path):
    """그림 글자는 ASCII 만(한글 폰트 없음)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, rows in result["curves"].items():
        ax.plot([r["latency_ms_pi"] for r in rows], [r["test_macro_f1"] for r in rows], ".-", ms=3,
                label=f"{name} (t2={result['t2_ms_cpu1'][name]:.3f} ms)")
    ax.axvline(result["t1_ms_cpu1"], color="gray", ls=":", label="Stage 1 only")
    ax.set(xlabel="Expected latency at 3.3% attacks, CPU batch 1 (ms)", ylabel="Test Macro-F1")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=Path, required=True, help="T1 _s2bal tradeoff_probs npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--latency-requests", type=int, default=2000)
    parser.add_argument("--limit", type=int, help="스모크용 행 수 제한(저장하지 않는다)")
    args = parser.parse_args()
    out = PROJECT_ROOT / "experiments/results/stage2_candidates_srbh_4class.json"
    fig = PROJECT_ROOT / "docs/figures/cascade/stage2_candidates_srbh_4class.png"
    try:
        if not args.limit and (out.exists() or fig.exists()):
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {out.name}")
        result = analyze(args)
        if not args.limit:
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
            save_figure(result, fig)
        print(json.dumps({k: result[k] for k in ("t1_ms_cpu1", "t2_ms_cpu1", "stage1_test_macro_f1",
                                                 "stage2_alone", "operating_point", "H-T3")},
                         ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"T3 2차 후보 비교 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
