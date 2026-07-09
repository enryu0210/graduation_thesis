"""
Phase 5 (RQ2) — 회피 공격 실행 & ASR 산출 (CLI)

무엇을 하나:
    1) 대상 탐지기를 clean train 으로 학습하고, clean test 에서 baseline 을 잡는다.
    2) problem_space 의 의미보존 변형을 test 공격 샘플에 적용해, **같은 전처리→예측 경로**로
       통과시킨다(입력만 오염, 파이프라인 동일 — docs/05 §6).
    3) 회피 성공률(ASR)을 benign-evasion(공격→Normal 예측) 기준으로 측정한다.
         · 단일 기법별 ASR (어떤 회피가 잘 통하는가)
         · 예산(k) 조합 ASR 곡선 (회피를 겹칠수록 얼마나 뚫리나)

지표 정의(docs/05 §2.2) — 둘을 반드시 함께 보고한다:
    주 지표 = benign-evasion = 변형된 공격이 Normal 로 예측된 비율(= WAF 우회, 유일하게 위험한 실패).
    보조 지표 = any-misclassification = 예측≠진짜 클래스 비율(공격 클래스 사이 '동요').
    주 지표가 0 이어도 보조 지표로 강건성 차이가 드러난다. 이번엔 CPU 로 되는 TF-IDF 대상.
    CNN 은 GPU 확보 후 동일 스크립트로.

산출물:
    experiments/results/evasion_{track}_tfidf_{clf}_single.json / _stacked.json
    docs/figures/attacks/evasion_single_{track}_tfidf_{clf}.png (기법별 ASR)
    docs/figures/attacks/evasion_stacked_{track}_tfidf_{clf}.png (예산-ASR 곡선)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("src/models", "src/eval", "src/attacks"):
    sys.path.insert(0, str(PROJECT_ROOT / extra))

from data_text import load_text_split, build_label_encoding, encode_labels_with  # noqa: E402
from baseline_tfidf import build_classifier  # noqa: E402
from metrics import _find_normal_index  # noqa: E402
import problem_space as PS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "attacks"


def train_target(track: str, clf_name: str, max_features: int):
    """clean train 으로 TF-IDF+분류기를 학습해 (vectorizer, clf, classes) 반환."""
    tr_txt, tr_lab = load_text_split(track, "train", "raw")
    y_train, classes = build_label_encoding(tr_lab)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          max_features=max_features, lowercase=False)
    Xtr = vec.fit_transform(tr_txt)
    clf = build_classifier(clf_name, len(classes))
    clf.fit(Xtr, y_train)
    return vec, clf, classes


def benign_evasion(pred: np.ndarray, mask: np.ndarray, normal_idx: int) -> float:
    """mask(적용 대상) 샘플 중 Normal 로 예측된 비율(= 주 지표 회피 성공률, docs/05 §2.2 ①)."""
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == normal_idx).mean())


def any_misclass(pred: np.ndarray, mask: np.ndarray, y_true: np.ndarray) -> float:
    """mask 샘플 중 '예측 != 진짜 클래스' 비율(= 보조 지표, docs/05 §2.2 ②).

    주 지표(benign-evasion)가 0 이어도 탐지기가 공격 클래스 사이에서 얼마나 흔들리는지
    보여준다. WAF 우회는 아니지만 표현방식별 강건성 비교의 핵심 렌즈다.
    """
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] != y_true[mask]).mean())


def run_single(vec, clf, classes, te_txt, te_lab, y_true, base_pred, normal_idx, seed):
    """단일 기법별 ASR 을 계산한다. 반환: rows(list of dict)."""
    rows = []
    for tech in PS.ALL_TECHNIQUES:
        mutated, applied = PS.mutate_single(te_txt, te_lab, tech, random.Random(seed))
        pred = clf.predict(vec.transform(mutated))
        rows.append({
            "technique": tech,
            "n_applied": int(applied.sum()),
            "asr_clean": benign_evasion(base_pred, applied, normal_idx),   # 주 지표 변형 전
            "asr_mutated": benign_evasion(pred, applied, normal_idx),      # 주 지표 변형 후
            # 보조 지표(any-misclassification) — 변형 전/후를 함께 기록해 '동요'를 정량화
            "anymis_clean": any_misclass(base_pred, applied, y_true),
            "anymis_mutated": any_misclass(pred, applied, y_true),
        })
        rows[-1]["asr_delta"] = rows[-1]["asr_mutated"] - rows[-1]["asr_clean"]
        rows[-1]["anymis_delta"] = rows[-1]["anymis_mutated"] - rows[-1]["anymis_clean"]
    return rows


def run_stacked(vec, clf, classes, te_txt, te_lab, y_true, base_pred, normal_idx, budget, seed):
    """예산 k=1..budget 조합 ASR 곡선을 계산한다(주·보조 지표 동시)."""
    attack_mask = np.asarray([bool(l in PS.ATTACK_LABELS) for l in te_lab])
    rows = []
    for k in range(1, budget + 1):
        mutated, applied = PS.mutate_stacked(te_txt, te_lab, k, random.Random(seed))
        pred = clf.predict(vec.transform(mutated))
        rows.append({
            "budget_k": k,
            "n_applied": int(applied.sum()),
            "asr_mutated": benign_evasion(pred, applied, normal_idx),
            "anymis_mutated": any_misclass(pred, applied, y_true),
        })
    # k=0(=clean) 기준점도 앞에 붙여 곡선이 baseline 에서 출발하게 한다.
    rows.insert(0, {"budget_k": 0, "n_applied": int(attack_mask.sum()),
                    "asr_mutated": benign_evasion(base_pred, attack_mask, normal_idx),
                    "anymis_mutated": any_misclass(base_pred, attack_mask, y_true)})
    return rows


def fig_single(rows, out_path: Path, title: str, clean_ref: float):
    """기법별 ASR 가로 막대(변형 후), clean 기준선 표시."""
    rows_sorted = sorted(rows, key=lambda r: r["asr_mutated"])
    names = [r["technique"] for r in rows_sorted]
    asr = [r["asr_mutated"] for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, asr, color="#c0392b")
    ax.axvline(clean_ref, ls="--", color="gray", lw=1, label=f"clean baseline = {clean_ref:.3f}")
    ax.set_xlabel("ASR (benign-evasion: attack predicted as Normal)")
    ax.set_xlim(0, max(0.05, max(asr) * 1.15))
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


def fig_stacked(rows, out_path: Path, title: str):
    """예산 k vs 지표 곡선 — 주 지표(benign-evasion)와 보조 지표(any-misclass)를 함께."""
    x = [r["budget_k"] for r in rows]
    y_be = [r["asr_mutated"] for r in rows]
    y_am = [r["anymis_mutated"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y_be, "o-", color="#c0392b", lw=2, label="benign-evasion (primary: attack->Normal)")
    ax.plot(x, y_am, "s--", color="#2980b9", lw=2, label="any-misclass (secondary: pred!=true)")
    ax.set_xlabel("stacked mutation budget k")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(x)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description="RQ2 회피 공격 ASR 산출(TF-IDF 대상)")
    p.add_argument("--track", default="payload_4class_csicnorm",
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"])
    p.add_argument("--clf", default="logreg", choices=["logreg", "rf"])
    p.add_argument("--budget", type=int, default=5, help="조합 공격 최대 예산 k")
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== RQ2 회피 ASR: track={args.track} clf={args.clf} ===")
    t0 = time.perf_counter()
    vec, clf, classes = train_target(args.track, args.clf, args.max_features)
    normal_idx = _find_normal_index(classes)
    print(f"  학습 완료({time.perf_counter()-t0:.1f}s) classes={classes} normal_idx={normal_idx}")

    te_txt, te_lab = load_text_split(args.track, "test", "raw")
    te_lab = list(te_lab)
    # 진짜 클래스 인덱스(보조 지표 any-misclass 계산에 필요). classes 순서에 맞춘다.
    lab2idx = {c: i for i, c in enumerate(classes)}
    y_true = np.asarray([lab2idx[l] for l in te_lab])
    base_pred = clf.predict(vec.transform(te_txt))

    # 단일 기법별 ASR (주: benign-evasion / 보조: any-misclass)
    single = run_single(vec, clf, classes, te_txt, te_lab, y_true, base_pred, normal_idx, args.seed)
    print("  [단일 기법별 — 주(benign-evasion) / 보조(any-misclass), 변형 후]")
    for r in sorted(single, key=lambda x: -x["anymis_mutated"]):
        print(f"    {r['technique']:<22} n={r['n_applied']:>6,}  "
              f"BE {r['asr_clean']:.4f}→{r['asr_mutated']:.4f} | "
              f"AM {r['anymis_clean']:.4f}→{r['anymis_mutated']:.4f} (Δ+{r['anymis_delta']:.4f})")

    # 예산 조합 곡선
    stacked = run_stacked(vec, clf, classes, te_txt, te_lab, y_true, base_pred, normal_idx,
                          args.budget, args.seed)
    print("  [예산 곡선 — benign-evasion / any-misclass]")
    for r in stacked:
        print(f"    k={r['budget_k']}  BE={r['asr_mutated']:.4f}  AM={r['anymis_mutated']:.4f}")

    # 저장
    tag = f"{args.track}_tfidf_{args.clf}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"evasion_{tag}_single.json", "w", encoding="utf-8") as f:
        json.dump({"track": args.track, "clf": args.clf, "classes": classes,
                   "results": single}, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / f"evasion_{tag}_stacked.json", "w", encoding="utf-8") as f:
        json.dump({"track": args.track, "clf": args.clf, "budget": args.budget,
                   "results": stacked}, f, ensure_ascii=False, indent=2)

    clean_ref = stacked[0]["asr_mutated"]  # k=0 기준(전체 공격 clean benign-evasion)
    fig_single(single, FIG_DIR / f"evasion_single_{tag}.png",
               f"RQ2 single-technique ASR — {args.clf} ({args.track})", clean_ref)
    fig_stacked(stacked, FIG_DIR / f"evasion_stacked_{tag}.png",
                f"RQ2 stacked-budget ASR — {args.clf} ({args.track})")
    print(f"\n[저장] 결과 → experiments/results/evasion_{tag}_*.json")
    print(f"[저장] 그림 → docs/figures/attacks/evasion_*_{tag}.png")


if __name__ == "__main__":
    main()
