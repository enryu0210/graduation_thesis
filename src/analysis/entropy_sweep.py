"""
Phase 6 (RQ4a) — 표현 은닉 스윕: '내용 기반 탐지가 어디서 붕괴하는가'를 실측

무엇을 하나:
    페이로드를 S0(평문)→S4(AES 랜덤IV)까지 단계적으로 가린 뒤, 각 단계에서 세 축을 잰다.
      ① 엔트로피(bits/byte)       — 표현이 얼마나 '내용을 감췄나'의 물리량
      ② 클래스 분리도(mean-image L2) — 이미지 공간에서 4클래스가 얼마나 구분되나
      ③ 재학습 탐지 성능(Macro-F1 / benign-evasion) — 그 표현으로 **재학습한** 탐지기의 실력
    ③에서 '재학습'이 핵심이다: 가역 인코딩(S1/S2)은 재학습하면 되살아나지만,
    랜덤 암호화(S4)는 재학습해도 배울 게 없어 랜덤으로 붕괴한다 → 그게 진짜 탐지 경계.

왜 CNN 이 아니라 TF-IDF 로 ③을 재나(지금 단계):
    GPU 없이 CPU 로 '내용 기반 탐지기'의 붕괴를 보이는 게 목적. TF-IDF(char n-gram)도
    '내용(바이트 패턴)'에 의존하는 탐지기라 경계를 동일하게 드러낸다. CNN 은 GPU 확보 후
    같은 축에 얹으면 된다(같은 스윕·같은 지표).

산출물:
    experiments/results/rq4a_entropy_sweep_{track}.json
    docs/figures/rq4/entropy_sweep_{track}.png  (엔트로피·분리도·F1 3축 곡선)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("src/models", "src/eval", "src/imaging", "src/attacks"):
    sys.path.insert(0, str(PROJECT_ROOT / extra))

from data_text import load_text_split, build_label_encoding, encode_labels_with  # noqa: E402
import encoding as E  # noqa: E402
import metrics as M  # noqa: E402
from payload_to_image import bytes_to_image, DEFAULT_SIDE  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "rq4"


def stratified_subsample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """클래스별로 최대 per_class 개씩 뽑은 인덱스를 반환한다(파일럿 속도용)."""
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        picked.append(idx)
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def class_separability(images: np.ndarray, y: np.ndarray, class_names: list[str]) -> float:
    """클래스 평균 이미지들 사이의 평균 L2 거리(클수록 클래스가 구분됨).

    AES 처럼 표현이 난수화되면 모든 클래스 평균이 균일값(~127.5)으로 수렴 → 거리 ≈ 0.
    """
    means = []
    for c in range(len(class_names)):
        mask = y == c
        if mask.sum() == 0:
            continue
        means.append(images[mask].reshape(mask.sum(), -1).mean(axis=0))
    if len(means) < 2:
        return 0.0
    dists = [np.linalg.norm(a - b) for a, b in combinations(means, 2)]
    return float(np.mean(dists))


def evaluate_stage(fn, train_texts, y_train, test_texts, y_test, class_names,
                   side: int, max_features: int) -> dict:
    """한 은닉 단계에 대해 3축(엔트로피·분리도·재학습 탐지)을 측정한다."""
    # 변환: 각 페이로드 → 관측 바이트열
    train_bytes = [fn(t) for t in train_texts]
    test_bytes = [fn(t) for t in test_texts]

    # ① 엔트로피(test 평균)
    entropy = float(np.mean([E.byte_entropy(b) for b in test_bytes]))

    # ② 클래스 분리도(test 이미지 기반)
    test_images = np.stack([bytes_to_image(b, side=side).astype(np.float32) for b in test_bytes])
    separability = class_separability(test_images, y_test, class_names)

    # ③ 재학습 탐지: 바이트를 latin-1 문자열로 → char n-gram TF-IDF + logreg 재학습
    train_str = [E.bytes_to_latin1(b) for b in train_bytes]
    test_str = [E.bytes_to_latin1(b) for b in test_bytes]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          max_features=max_features, lowercase=False)
    Xtr = vec.fit_transform(train_str)
    Xte = vec.transform(test_str)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    m = M.compute_metrics(y_test, y_pred, class_names)

    af = m.get("attack_focused", {})
    return {
        "entropy_bits_per_byte": entropy,
        "class_separability_l2": separability,
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "attack_detection_recall": af.get("attack_detection_recall"),
        "benign_evasion_rate": af.get("benign_evasion_rate"),
    }


def save_figure(codes, names, rows, out_path: Path, random_f1: float) -> Path:
    """엔트로피·Macro-F1 2축 곡선(트윈 축).

    라벨은 영문으로 둔다(matplotlib 기본 폰트에 한글이 없어 깨짐 방지). 축 코드는
    ASCII 인 stage 코드(S0_raw…)를 쓴다. 분리도는 페이로드 길이에 교란되므로(암호문
    길이가 클래스마다 달라 mean-image 에 새어듦) 헤드라인 그림에서는 빼고 JSON 에만 남긴다.
    """
    x = np.arange(len(codes))
    ent = [r["entropy_bits_per_byte"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, f1, "o-", color="#c0392b", lw=2, label="Macro-F1 (retrained detector)")
    ax1.axhline(random_f1, ls="--", color="gray", lw=1,
                label=f"random baseline ~ {random_f1:.2f}")
    ax1.set_ylabel("Macro-F1", color="#c0392b")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels(codes, rotation=15)

    ax2 = ax1.twinx()
    ax2.plot(x, ent, "s--", color="#2980b9", lw=1.5, label="entropy (bits/byte)")
    ax2.set_ylabel("entropy (bits/byte)", color="#2980b9")
    ax2.set_ylim(0, 8.4)

    # S4(랜덤 암호화)에서의 붕괴를 화살표로 강조
    ax1.annotate("content detection\ncollapses under\nrandomized encryption",
                 xy=(len(codes) - 1, f1[-1]), xytext=(len(codes) - 2.3, 0.55),
                 fontsize=8, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b"))

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)
    ax1.set_title("RQ4a: encoding survives retraining, "
                  "randomized encryption breaks content detection", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description="RQ4a 표현 은닉 스윕(엔트로피·분리도·재학습 탐지)")
    p.add_argument("--track", default="payload_4class",
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"])
    p.add_argument("--per-class", type=int, default=4000,
                   help="클래스별 표본 상한(파일럿 속도용; 0=전체)")
    p.add_argument("--side", type=int, default=DEFAULT_SIDE)
    p.add_argument("--max-features", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== RQ4a 은닉 스윕: track={args.track} per_class={args.per_class} ===")
    tr_txt, tr_lab = load_text_split(args.track, "train", "raw")
    te_txt, te_lab = load_text_split(args.track, "test", "raw")
    y_train_all, classes = build_label_encoding(tr_lab)
    y_test_all = encode_labels_with(te_lab, classes)
    tr_txt = np.array(tr_txt, dtype=object)
    te_txt = np.array(te_txt, dtype=object)

    # 파일럿: 클래스별 표본 상한으로 다운샘플(속도). 0 이면 전체 사용.
    if args.per_class > 0:
        tr_sel = stratified_subsample(y_train_all, args.per_class, args.seed)
        te_sel = stratified_subsample(y_test_all, args.per_class, args.seed)
        tr_txt, y_train = tr_txt[tr_sel], y_train_all[tr_sel]
        te_txt, y_test = te_txt[te_sel], y_test_all[te_sel]
    else:
        y_train, y_test = y_train_all, y_test_all
    print(f"  train={len(tr_txt):,} test={len(te_txt):,} classes={classes}")

    rows, codes, names = [], [], []
    for code, name, fn, rev in E.STAGES:
        t0 = time.perf_counter()
        r = evaluate_stage(fn, list(tr_txt), y_train, list(te_txt), y_test,
                           classes, args.side, args.max_features)
        r.update({"stage": code, "reversibility": rev})
        rows.append(r); codes.append(code); names.append(name)
        print(f"  {code:<14} 엔트로피={r['entropy_bits_per_byte']:.2f}  "
              f"분리도={r['class_separability_l2']:.1f}  "
              f"MacroF1={r['macro_f1']:.4f}  benign-evasion="
              f"{r['benign_evasion_rate'] if r['benign_evasion_rate'] is not None else float('nan'):.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

    random_f1 = 1.0 / len(classes)  # 무정보 분류기의 대략적 기준선
    out_json = RESULTS_DIR / f"rq4a_entropy_sweep_{args.track}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"track": args.track, "classes": classes,
                   "random_f1_baseline": random_f1, "stages": rows},
                  f, ensure_ascii=False, indent=2)
    fig = save_figure(codes, names, rows, FIG_DIR / f"entropy_sweep_{args.track}.png", random_f1)
    print(f"\n[저장] 결과 → {out_json.relative_to(PROJECT_ROOT)}")
    print(f"[저장] 곡선 → {fig.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
