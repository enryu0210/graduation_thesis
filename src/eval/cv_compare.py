"""
Phase 4/RQ1 보강 — 교차검증 결과 비교 + paired t-test (유의성 판정)

왜 필요한가:
    cross_validate.py 가 만든 fold 별 지표를 모아, "설정 A 가 설정 B 보다 낫다"가
    **통계적으로 지지되는지** 판정한다. 단일 split 로는 실행 간 변동(±0.11pp, cuDNN
    비결정성)이 설정 간 차이보다 커서 순위를 믿을 수 없었다(docs/04 §5 재현성 메모).

왜 paired(대응표본) t-test 인가:
    모든 설정이 **같은 fold 분할**을 공유하므로(cross_validate.py 가 (라벨, seed)로만
    fold 를 정함), fold 난이도라는 공통 변동을 상쇄하고 "설정 차이"만 볼 수 있다.
    독립표본 t-test 를 쓰면 fold 난이도 편차에 묻혀 검정력이 크게 떨어진다.
    → label_fingerprint 가 모든 설정에서 같은지 먼저 검증한다(다르면 대응 관계가 깨짐).

⚠️ 해석 주의:
    - fold=5 라 자유도가 4뿐이다. 검정력이 낮아 "유의하지 않음"이 곧 "차이 없음"은 아니다.
    - 쌍별 비교를 여러 번 하면 우연히 유의해질 확률이 커진다 → Holm-Bonferroni 보정을 적용한다.

사용법:
    python src/eval/cv_compare.py                     # MCC 기준 전체 비교
    python src/eval/cv_compare.py --metric macro_f1
    python src/eval/cv_compare.py --ref payload_4class_cnn_raw_rgb   # 특정 설정 기준 비교
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "models"

# 그림에 쓸 짧은 이름(matplotlib 은 ASCII 만 — DejaVu Sans 에 한글 없음).
def short_name(tag: str) -> str:
    return tag.replace("payload_4class_", "").replace("_raw", "")


def load_cv_results(metric: str) -> tuple[dict[str, np.ndarray], str]:
    """cv_*.json 을 모아 {tag: fold별 지표 배열} 로 만든다. fold 정렬 검증 포함."""
    runs: dict[str, np.ndarray] = {}
    fingerprints: dict[str, str] = {}

    for path in sorted(glob.glob(str(RESULTS_DIR / "cv_*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 스모크(folds<5) 산출물이 섞이면 비교가 오염된다 → 걸러낸다.
        if data.get("folds", 0) < 5:
            print(f"  [건너뜀] {Path(path).name} (folds={data.get('folds')}, 스모크 산출물)")
            continue

        summary = data.get("summary", {})
        if metric not in summary:
            print(f"  [건너뜀] {Path(path).name} ({metric} 없음)")
            continue

        tag = data["tag"]
        runs[tag] = np.asarray(summary[metric]["per_fold"], dtype=float)
        fingerprints[tag] = data.get("label_fingerprint", "?")

    if not runs:
        raise SystemExit("비교할 CV 결과가 없습니다. 먼저 cross_validate.py 를 실행하세요.")

    # 대응표본의 전제: 모든 설정이 같은 풀·같은 fold 를 봤는가?
    unique_fp = set(fingerprints.values())
    if len(unique_fp) > 1:
        raise SystemExit(
            "라벨 지문이 서로 다릅니다 → fold 대응 관계가 깨져 paired 비교를 할 수 없습니다.\n"
            + "\n".join(f"  {t}: {f}" for t, f in fingerprints.items())
        )
    return runs, unique_fp.pop()


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni 보정. 쌍별 비교를 여러 번 할 때의 다중비교 문제를 완화한다.

    Bonferroni 보다 덜 보수적(검정력 손실이 작음)이면서 family-wise error 를 통제한다.
    반환: 입력과 같은 순서의 보정 p-value.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # (m - rank) 를 곱하고, 순서가 뒤집히지 않도록 누적 최대값을 유지한다.
        value = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def plot_ranking(runs: dict[str, np.ndarray], metric: str, out_path: Path) -> None:
    """평균±표준편차 막대그래프. 라벨은 ASCII 만(한글은 □ 로 깨짐 — CLAUDE.md)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(runs.items(), key=lambda kv: kv[1].mean(), reverse=True)
    names = [short_name(t) for t, _ in items]
    means = [v.mean() for _, v in items]
    stds = [v.std(ddof=1) for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(items) + 2))
    ax.barh(range(len(items)), means, xerr=stds, capsize=4, color="#4C72B0")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} (5-fold mean +- SD)")
    ax.set_title(f"RQ1 5-fold cross-validation: {metric}")
    # 값 차이가 작아 0부터 그리면 구분이 안 된다 → 관심 구간만 확대.
    lo = min(m - s for m, s in zip(means, stds))
    ax.set_xlim(max(0.0, lo - 0.01), 1.0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="CV 결과 비교 + paired t-test")
    parser.add_argument("--metric", default="mcc",
                        choices=["mcc", "macro_f1", "accuracy", "pr_auc_macro",
                                 "benign_evasion_rate"])
    parser.add_argument("--ref", default=None,
                        help="이 설정을 기준으로만 비교(생략 시 전체 쌍 비교)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    from scipy import stats

    runs, fingerprint = load_cv_results(args.metric)
    print(f"\n=== RQ1 5-fold CV 비교: {args.metric} (설정 {len(runs)}개, 라벨지문 {fingerprint}) ===\n")

    # 1) 순위표
    ranked = sorted(runs.items(), key=lambda kv: kv[1].mean(), reverse=True)
    print(f"{'설정':40}{'평균':>9}{'표준편차':>11}   fold별")
    for tag, vals in ranked:
        folds = " ".join(f"{v:.4f}" for v in vals)
        print(f"{short_name(tag):40}{vals.mean():>9.4f}{vals.std(ddof=1):>11.4f}   {folds}")

    # 2) 쌍별 paired t-test
    if args.ref:
        if args.ref not in runs:
            raise SystemExit(f"--ref 를 찾을 수 없습니다: {args.ref}\n후보: {list(runs)}")
        pairs = [(args.ref, t) for t in runs if t != args.ref]
    else:
        pairs = list(combinations([t for t, _ in ranked], 2))

    raw_p, diffs = [], []
    for a, b in pairs:
        diff = runs[a] - runs[b]
        # 모든 fold 에서 값이 완전히 같으면 t-test 가 정의되지 않는다(0으로 나눔).
        if np.allclose(diff, 0):
            raw_p.append(1.0)
        else:
            raw_p.append(float(stats.ttest_rel(runs[a], runs[b]).pvalue))
        diffs.append(float(diff.mean()))

    adj_p = holm_bonferroni(raw_p)

    print(f"\n--- paired t-test (fold=5, df=4, Holm-Bonferroni 보정, alpha={args.alpha}) ---")
    print(f"{'A vs B':58}{'평균차(A-B)':>13}{'p(raw)':>10}{'p(adj)':>10}  판정")
    for (a, b), d, p, pa in sorted(zip(pairs, diffs, raw_p, adj_p), key=lambda z: z[3]):
        verdict = "유의" if pa < args.alpha else "판정불가"
        print(f"{short_name(a) + ' vs ' + short_name(b):58}{d:>+13.4f}{p:>10.4f}{pa:>10.4f}  {verdict}")

    print("\n⚠️ fold=5 는 자유도가 4뿐이라 검정력이 낮다. '판정불가'는 '차이 없음'이 아니라")
    print("   '이 표본수로는 구분 못 함'을 뜻한다(추가 fold/반복이 필요).")

    out = FIG_DIR / f"cv_ranking_{args.metric}.png"
    plot_ranking(runs, args.metric, out)
    print(f"\n[그림] {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
