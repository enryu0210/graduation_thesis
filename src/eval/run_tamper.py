"""Phase 12 (M1) — 표현 불일치 기반 '회피 시도' 탐지 실행 (CLI)

무엇을 하나:
    캐스케이드(1차 RGB CNN + 2차 char-CNN)가 이미 계산하는 두 확률분포의 불일치로
    "이 요청이 탐지 회피를 시도했는가"를 판별하고, 판정 기준(docs/11 §7)에 따라 채점한다.
    **새 학습이 전혀 없다** — 기존 체크포인트와 기존 변형 생성기를 그대로 재사용한다.

    같은 실행에서 G3(변형 조건 상보성, docs/11 §3.4)도 함께 해결한다. clean 에서 순증이
    0 이었던 상보성 표(F4)가 변형 하에서 어떻게 바뀌는지가 M1 서술의 직접 근거다.

실험 설계 — 무엇을 무엇과 구분하나:
    양성 = 변형된 공격 페이로드(problem_space 의 의미보존 변형, 예산 k)
    음성 = **같은 공격 페이로드의 clean 판본**(paired)
    → 음성으로 정상 트래픽(Normal)을 쓰면 "공격 vs 정상"을 푸는 셈이라 과제가 쉬워진다.
      가장 엄격한 대조(같은 샘플의 변형 전/후)를 주 판정으로 삼고, 정상 트래픽 대비는
      보조로 함께 기록한다. 이 선택은 **실측 전에** 고정한 것이다(HARKing 방지).

두 운영 모드(docs/11 §3.2) — 분리해 보고한다:
    모드 A(비용 불변) : 에스컬레이션된 샘플에만 판정. 2차를 어차피 돌리므로 **추가 비용 0**.
                       M1 이 주장하는 기여는 이 모드 하나뿐이다(docs/12 §4.2).
    모드 B(전량)      : 전량에 2차를 돌린 상한. 비용이 융합형과 같아지므로 참고용.

방법론:
    · τ 는 cascade.py 가 clean val 에서 확정한 값을 **읽어서** 쓴다(공격 데이터로 재튜닝 금지).
    · tampering 임계값도 **clean val 에서 확정 → test 에 1회 적용**한다.
    · 판정(H5-1/H5-2)은 tamper.verdict 가 자동으로 찍는다(사후 조정 여지 제거).

산출물:
    experiments/results/tamper_{cascade_tag}.json
    docs/figures/attacks/tamper_auc_{tag}.png       — 예산 k vs ROC-AUC (점수함수 3종 × 모드 2종)
    docs/figures/attacks/tamper_coverage_{tag}.png  — τ 스윕: 모드 A 커버리지 vs 판별력

사용법 (GPU 권장 — 캐스케이드 체크포인트와 cascade.py 리포트가 먼저 있어야 함):
    python src/eval/run_tamper.py --track payload_4class_csicnorm --channels rgb
    python src/eval/run_tamper.py --track payload_4class_csicnorm --channels rgb --smoke
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("src/models", "src/eval", "src/attacks", "src/imaging", "src/data"):
    sys.path.insert(0, str(PROJECT_ROOT / extra))

import tamper as T  # noqa: E402
import data_image  # noqa: E402  (채널 접미사 규칙 재사용 — tag 일치)
import problem_space as PS  # noqa: E402
from data_text import load_text_split  # noqa: E402
from metrics import _find_normal_index  # noqa: E402
from tagging import DEFAULT_AUG_RATIO, DEFENSE_MODES, defense_suffix  # noqa: E402
# 모델 로딩·변형 텍스트 예측 경로는 run_evasion 의 것을 그대로 재사용한다.
# 왜: "같은 공격, 같은 전처리 경로"가 아니면 회피 실험(RQ2/RQ3)의 수치와 비교가 불가능하다.
from run_evasion import build_torch_proba, load_cascade_tau  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "attacks"

MODES = ("mode_a", "mode_b")


# ---------------------------------------------------------------------------
# 확률 계산
# ---------------------------------------------------------------------------
def stage_probs(proba1, proba2, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """한 조건(clean 또는 변형 k)에 대해 1·2차 확률을 모두 계산한다.

    ⚠️ 여기서 2차를 **전량** 돌리는 것은 평가 편의다(모드 B 상한을 재려면 필요하다).
    배포 시 모드 A 는 에스컬레이션분만 쓰므로 추가 비용이 0 이라는 주장과 모순되지 않는다 —
    모드 A 의 지표는 에스컬레이션 마스크로 걸러낸 부분집합에서만 계산한다.
    """
    return proba1(texts), proba2(texts)


def mode_population(p1: np.ndarray, mode: str, tau: float) -> np.ndarray:
    """모드별 '판정 가능한 샘플' 마스크."""
    if mode == "mode_a":
        return T.escalated_mask(p1, tau)
    return np.ones(len(p1), dtype=bool)  # 모드 B: 전량


# ---------------------------------------------------------------------------
# 평가 본체
# ---------------------------------------------------------------------------
def evaluate_scorer(scorer: str, mode: str, tau: float, target_fpr: float,
                    val: dict, clean: dict, mutated: list[dict],
                    attack_mask: np.ndarray, normal_mask: np.ndarray) -> dict:
    """(점수함수, 모드) 한 조합의 전체 결과를 낸다.

    val 에서 임계값을 확정한 뒤 test 의 clean/변형 조건에 적용한다.
    """
    # 1) 임계값 — clean val 의 **정상 트래픽** 점수 분포에서 목표 오탐률로 확정한다.
    #    H5-2 의 기준이 "clean Normal 오탐 ≤ 1%" 이므로 그 집합에서 직접 고르는 것이 맞다.
    s_val = T.tampering_scores(val["p1"], val["p2"], scorer)
    pop_val = mode_population(val["p1"], mode, tau)
    thr = T.select_threshold(s_val[pop_val & val["normal"]], target_fpr)

    # 2) clean test — 음성 집합
    s_clean = T.tampering_scores(clean["p1"], clean["p2"], scorer)
    pop_clean = mode_population(clean["p1"], mode, tau)
    neg_paired = s_clean[pop_clean & attack_mask]      # 주 대조: 같은 공격의 변형 전
    neg_normal = s_clean[pop_clean & normal_mask]      # 보조 대조: 정상 트래픽

    rows = []
    for cond in mutated:
        s_mut = T.tampering_scores(cond["p1"], cond["p2"], scorer)
        pop_mut = mode_population(cond["p1"], mode, tau)
        pos = s_mut[pop_mut & attack_mask]

        row = {"budget_k": cond["k"]}
        row.update(T.binary_eval(neg_paired, pos, thr))
        # 정상 트래픽을 음성으로 둔 보조 판별(운영 관점: "정상과 구분되는가")
        aux = T.binary_eval(neg_normal, pos, thr)
        row["roc_auc_vs_normal"] = aux["roc_auc"]
        # 커버리지 = 전체 대비 판정 가능 비율. 모드 A 의 실용성을 좌우한다(H5-2).
        row["coverage_positive"] = float(pop_mut[attack_mask].mean())
        rows.append(row)

    return {
        "threshold": thr,
        "threshold_source": f"clean val Normal, target_fpr={target_fpr:g}",
        "coverage_clean_attack": float(pop_clean[attack_mask].mean()),
        "coverage_clean_normal": float(pop_clean[normal_mask].mean()),
        "clean_normal_fpr": float(T.flag(neg_normal, thr).mean()) if len(neg_normal) else None,
        "clean_attack_fpr": float(T.flag(neg_paired, thr).mean()) if len(neg_paired) else None,
        "by_budget": rows,
    }


def tau_sweep(scorer: str, taus: np.ndarray, clean: dict, mutated_last: dict,
              attack_mask: np.ndarray) -> list[dict]:
    """τ 를 움직이며 모드 A 의 (커버리지, 판별력)을 그린다.

    왜 이 곡선이 필요한가(docs/11 §9): 모드 A 는 에스컬레이션된 샘플에만 정의되므로,
    운영 τ 가 낮으면 커버리지가 낮아 실용성이 떨어진다. τ 를 올리면 커버리지가 오르지만
    2차 호출이 늘어 비용도 오른다 → **τ 에 "탐지 커버리지"라는 새 해석 축**이 생긴다.
    확률은 이미 계산돼 있으므로 이 스윕의 추가 비용은 0 이다.
    """
    s_clean = T.tampering_scores(clean["p1"], clean["p2"], scorer)
    s_mut = T.tampering_scores(mutated_last["p1"], mutated_last["p2"], scorer)

    rows = []
    for tau in taus:
        pop_c = T.escalated_mask(clean["p1"], tau)
        pop_m = T.escalated_mask(mutated_last["p1"], tau)
        ev = T.binary_eval(s_clean[pop_c & attack_mask], s_mut[pop_m & attack_mask],
                           threshold=float("inf"))  # 곡선에서는 AUC 만 본다
        rows.append({
            "tau": float(tau),
            "coverage_clean": float(pop_c[attack_mask].mean()),
            "coverage_mutated": float(pop_m[attack_mask].mean()),
            "roc_auc": ev["roc_auc"],
            "n_negative": ev["n_negative"],
            "n_positive": ev["n_positive"],
        })
    return rows


# ---------------------------------------------------------------------------
# 그림 (라벨은 ASCII 만 — DejaVu Sans 에 한글이 없다)
# ---------------------------------------------------------------------------
def fig_auc(results: dict, out_path: Path, title: str) -> None:
    """예산 k vs ROC-AUC. 왼쪽=모드 A(비용 0), 오른쪽=모드 B(상한)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {"top1_disagree": "#7f8c8d", "js_divergence": "#c0392b", "conf_gap": "#2980b9"}

    for ax, mode, sub in zip(axes, MODES, ("mode A (escalated only, +0 cost)", "mode B (all, upper bound)")):
        for scorer, color in colors.items():
            rows = results[scorer][mode]["by_budget"]
            x = [r["budget_k"] for r in rows]
            y = [r["roc_auc"] if r["roc_auc"] is not None else np.nan for r in rows]
            style = "o-" if scorer == T.PRIMARY_SCORER else "s--"
            ax.plot(x, y, style, color=color, lw=2, ms=5,
                    label=scorer + (" (primary)" if scorer == T.PRIMARY_SCORER else ""))
        ax.axhline(0.90, ls=":", color="#27ae60", lw=1.2, label="H5-1 pass = 0.90")
        ax.axhline(0.75, ls=":", color="#e67e22", lw=1.2, label="H5-1 reject = 0.75")
        ax.set_xlabel("stacked mutation budget k")
        ax.set_xticks(sorted({r["budget_k"] for r in results[T.PRIMARY_SCORER][mode]["by_budget"]}))
        ax.set_title(sub, fontsize=11)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ROC-AUC (tampered vs same payload clean)")
    axes[0].legend(loc="lower right", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fig_coverage(rows: list[dict], op_coverage: float, op_auc: float | None,
                 out_path: Path, title: str) -> None:
    """τ 스윕: 모드 A 커버리지 vs 판별력. 운영점(현행 τ)을 함께 찍는다."""
    x = [r["coverage_mutated"] for r in rows]
    y = [r["roc_auc"] if r["roc_auc"] is not None else np.nan for r in rows]
    order = np.argsort(x)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(np.asarray(x)[order], np.asarray(y)[order], "o-", color="#8e44ad", lw=2, ms=4,
            label="mode A (tau sweep)")
    ax.axhline(0.90, ls=":", color="#27ae60", lw=1.2, label="H5-1 pass = 0.90")
    ax.axvline(0.30, ls=":", color="#e67e22", lw=1.2, label="H5-2 coverage = 0.30")
    if op_auc is not None:
        ax.plot([op_coverage], [op_auc], "D", color="#27ae60", ms=9,
                label=f"operating tau (coverage={op_coverage:.3f})")
    ax.set_xlabel("mode-A coverage (fraction of attacks escalated = judgeable)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(
        description="M1 — 캐스케이드의 표현 불일치로 회피 시도를 탐지(새 학습 없음)")
    p.add_argument("--track", default="payload_4class_csicnorm",
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"])
    p.add_argument("--stage2", default="charcnn", choices=["charcnn", "bilstm"],
                   help="2차 판정기(1차는 CNN 고정 — 캐스케이드 구성과 같아야 한다)")
    p.add_argument("--text", default="raw", choices=["raw", "decoded"])
    p.add_argument("--channels", default="gray", choices=["gray", "rgb"])
    p.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy")
    p.add_argument("--unbalanced", action="store_true",
                   help="'_bal' 없는 체크포인트를 사용(기본은 balanced)")
    p.add_argument("--side", type=int, default=48)
    p.add_argument("--max-len", type=int, default=48 * 48)
    p.add_argument("--budget", type=int, default=5, help="조합 변형 최대 예산 k")
    p.add_argument("--target-fpr", type=float, default=0.01,
                   help="tampering 임계값을 고를 때의 목표 오탐률(H5-2 기준 = 0.01)")
    p.add_argument("--tau", type=float, default=None,
                   help="캐스케이드 운영점 직접 지정. 기본은 cascade.py 가 val 에서 확정한 값")
    p.add_argument("--n-taus", type=int, default=21, help="τ 스윕 격자 개수(커버리지 곡선용)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="샘플 수 제한(스모크용)")
    p.add_argument("--smoke", action="store_true", help="빠른 동작 확인(작게, 저장 생략)")
    # 방어 축 — 방어 구성에서 불일치 신호가 약해지는지 확인해야 한다(docs/11 §9 리스크).
    p.add_argument("--defense", default="none", choices=list(DEFENSE_MODES))
    p.add_argument("--mutation-split", default=None, choices=["S0", "SA"])
    p.add_argument("--aug-ratio", type=float, default=None)
    args = p.parse_args()

    # 방어 축 정합성(cascade.py 와 같은 규칙 — 어긋난 체크포인트를 조용히 집는 사고 방지)
    if args.defense == "advtrain":
        if args.mutation_split is None:
            p.error("--defense advtrain 은 --mutation-split {S0,SA} 가 필요합니다")
        if args.aug_ratio is None:
            args.aug_ratio = DEFAULT_AUG_RATIO
    elif args.mutation_split is not None or args.aug_ratio is not None:
        p.error("--mutation-split/--aug-ratio 는 --defense advtrain 일 때만 의미가 있습니다")
    if args.defense == "norm" and args.text != "decoded":
        p.error("--defense norm 은 --text decoded 와 함께 써야 합니다")

    if args.smoke:
        args.limit = args.limit or 400
        args.budget = min(args.budget, 2)
        args.n_taus = min(args.n_taus, 7)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    balanced = not args.unbalanced
    print(f"=== M1 표현 불일치 탐지: track={args.track} channels={args.channels} "
          f"defense={args.defense} device={device} ===")
    t0 = time.perf_counter()

    # 1) 두 단계 모델 — run_evasion 과 같은 로더/전처리 경로
    proba1, classes = build_torch_proba(
        args.track, "cnn", args.side, args.max_len, balanced, device,
        text=args.text, channels=args.channels, encoders=encoders,
        defense=args.defense, mutation_split=args.mutation_split, aug_ratio=args.aug_ratio)
    proba2, classes2 = build_torch_proba(
        args.track, args.stage2, args.side, args.max_len, balanced, device,
        text=args.text, defense=args.defense,
        mutation_split=args.mutation_split, aug_ratio=args.aug_ratio)
    if classes != classes2:
        raise ValueError(f"두 단계의 클래스 순서가 다릅니다: {classes} vs {classes2}")
    normal_idx = _find_normal_index(classes)

    # 2) τ — 공격 데이터로 재튜닝하지 않는다(cascade.py 가 clean val 에서 확정한 값)
    tau = args.tau if args.tau is not None else load_cascade_tau(
        args.track, args.stage2, args.text, balanced, args.channels, encoders,
        args.defense, args.mutation_split, args.aug_ratio)
    print(f"  준비 완료({time.perf_counter()-t0:.1f}s) classes={classes} "
          f"normal_idx={normal_idx} tau={tau:.6f}")

    # 3) 입력 — 공격 입력은 항상 raw 다(정규화 방어는 '탐지 직전'에 proba 안에서 적용된다)
    va_txt, va_lab = load_text_split(args.track, "val", "raw")
    te_txt, te_lab = load_text_split(args.track, "test", "raw")
    va_txt, va_lab = list(va_txt), list(va_lab)
    te_txt, te_lab = list(te_txt), list(te_lab)
    if args.limit:
        va_txt, va_lab = va_txt[:args.limit], va_lab[:args.limit]
        te_txt, te_lab = te_txt[:args.limit], te_lab[:args.limit]

    lab2idx = {c: i for i, c in enumerate(classes)}
    y_te = np.asarray([lab2idx[l] for l in te_lab])
    normal_mask = y_te == normal_idx

    # 4) clean 확률 (val: 임계값용 / test: 음성 집합)
    p1_va, p2_va = stage_probs(proba1, proba2, va_txt)
    y_va = np.asarray([lab2idx[l] for l in va_lab])
    val = {"p1": p1_va, "p2": p2_va, "normal": y_va == normal_idx}
    p1_c, p2_c = stage_probs(proba1, proba2, te_txt)
    clean = {"p1": p1_c, "p2": p2_c}
    print(f"  clean 확률 계산 완료 val={len(va_txt):,} test={len(te_txt):,} "
          f"({time.perf_counter()-t0:.1f}s)")

    # 5) 변형 조건 k=1..budget. applied 는 '변형 대상 공격 샘플' 마스크(k 와 무관하게 동일).
    mutated, attack_mask = [], None
    for k in range(1, args.budget + 1):
        mut_txt, applied = PS.mutate_stacked(te_txt, te_lab, k, random.Random(args.seed))
        # applied 는 라벨에만 의존하므로 k 가 달라도 같아야 한다. 어긋나면 paired 대조가
        # 깨진 것이므로(음성/양성이 다른 샘플 집합이 된다) 조용히 넘어가지 않는다.
        if attack_mask is None:
            attack_mask = applied
        elif not np.array_equal(attack_mask, applied):
            raise ValueError(f"k={k} 에서 변형 대상 마스크가 달라졌습니다 — paired 대조가 깨집니다.")
        p1_k, p2_k = stage_probs(proba1, proba2, mut_txt)
        mutated.append({"k": k, "p1": p1_k, "p2": p2_k, "pred1": p1_k.argmax(axis=1),
                        "pred2": p2_k.argmax(axis=1)})
        print(f"  변형 k={k} 확률 계산 완료 (n_applied={int(applied.sum()):,}, "
              f"{time.perf_counter()-t0:.1f}s)")

    # 6) 점수함수 3종 × 모드 2종
    results = {}
    for scorer in T.SCORERS:
        results[scorer] = {}
        for mode in MODES:
            results[scorer][mode] = evaluate_scorer(
                scorer, mode, tau, args.target_fpr, val, clean, mutated,
                attack_mask, normal_mask)

    # 7) τ 스윕(모드 A 커버리지 해석 축) — 최대 예산 조건 기준
    conf_grid = np.quantile(p1_c.max(axis=1), np.linspace(0.0, 1.0, args.n_taus))
    taus = np.unique(np.concatenate([conf_grid, [tau, 1.01]]))
    sweep = tau_sweep(T.PRIMARY_SCORER, taus, clean, mutated[-1], attack_mask)

    # 8) G3 — 변형 조건 상보성(1차 이미지 표현 vs 2차 텍스트 표현)
    comp = {
        "clean": T.complementarity(y_te, p1_c.argmax(axis=1), p2_c.argmax(axis=1), normal_idx),
        f"mutated_k{args.budget}": T.complementarity(
            y_te, mutated[-1]["pred1"], mutated[-1]["pred2"], normal_idx),
        "note": ("a=stage1(image CNN), b=stage2(char-CNN). detection=Normal 로 새지 않음, "
                 "correct=진짜 클래스 일치. 변형 하 격차는 대부분 미탐이 아니라 오귀속이다."),
    }

    # 9) 판정 — 사전 고정 기준(docs/11 §7)을 코드가 그대로 적용한다
    primary = results[T.PRIMARY_SCORER][T.PRIMARY_MODE]
    last = primary["by_budget"][-1]
    verdict = T.verdict(last["roc_auc"], primary["clean_normal_fpr"],
                        last["coverage_positive"])

    # 10) 콘솔 요약
    print(f"\n  [주 지표] scorer={T.PRIMARY_SCORER} mode={T.PRIMARY_MODE} "
          f"threshold={primary['threshold']:.6f}")
    for r in primary["by_budget"]:
        auc = r["roc_auc"]
        print(f"    k={r['budget_k']}  AUC={auc if auc is None else f'{auc:.4f}'}  "
              f"TPR={r['tpr']:.4f}  coverage={r['coverage_positive']:.4f}  "
              f"(n+={r['n_positive']:,} n-={r['n_negative']:,})")
    print(f"    clean Normal FPR={primary['clean_normal_fpr']:.4f} / "
          f"clean attack FPR={primary['clean_attack_fpr']:.4f}")
    print(f"  [판정] H5-1={verdict['H5-1']['result']} / H5-2={verdict['H5-2']['result']}")
    print("  [ablation — 최대 예산 AUC]")
    for scorer in T.SCORERS:
        a = results[scorer]["mode_a"]["by_budget"][-1]["roc_auc"]
        b = results[scorer]["mode_b"]["by_budget"][-1]["roc_auc"]
        fmt = lambda v: "None" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"    {scorer:<15} mode A={fmt(a)}  mode B={fmt(b)}")
    cm = comp[f"mutated_k{args.budget}"]["correct"]
    print(f"  [G3 변형 상보성 k={args.budget}, correct 기준] "
          f"both={cm['both']:,} stage1만={cm['a_only']:,} stage2만={cm['b_only']:,} "
          f"둘다실패={cm['neither']:,}")

    if args.smoke:
        print("\n  [smoke] 저장 생략(코드 동작 확인만).")
        return

    # 11) 저장 — cascade.py 와 같은 축으로 tag 를 만든다(docs/11 §6: M1 은 새 모델 축이 없다).
    #     ⚠️ 실험을 가르는 값은 전부 tag 에 넣는다. 예산 k·목표 FPR 이 다르면 다른 실험이므로
    #        기본값이 아닐 때 접미사를 붙인다(train.py 의 _lr 관습 계승 — 기본값이면 생략).
    bal = "_bal" if balanced else ""
    s2 = "" if args.stage2 == "charcnn" else f"-{args.stage2}"
    ch_tag = data_image._channel_suffix(args.channels, encoders)
    text_tag = "" if args.text == "raw" else f"_{args.text}"
    def_tag = defense_suffix(args.defense, args.mutation_split, args.aug_ratio)
    tau_tag = f"_tau{args.tau:g}" if args.tau is not None else ""
    k_tag = "" if args.budget == 5 else f"_k{args.budget}"
    fpr_tag = "" if args.target_fpr == 0.01 else f"_fpr{args.target_fpr:g}"
    tag = (f"{args.track}_cascade{s2}{text_tag}{ch_tag}{def_tag}{tau_tag}"
           f"{k_tag}{fpr_tag}{bal}")

    payload = {
        "config": {
            "track": args.track, "stage1": "cnn", "stage2": args.stage2,
            "text": args.text, "channels": args.channels, "encoders": list(encoders),
            "balance": balanced, "tau": tau, "tau_source":
                "cli" if args.tau is not None else "cascade.py val 확정값",
            "budget": args.budget, "seed": args.seed, "target_fpr": args.target_fpr,
            "limit": args.limit, "device": str(device),
            "defense": {"mode": args.defense,
                        "mutation_split": args.mutation_split if args.defense == "advtrain" else None,
                        "aug_ratio": args.aug_ratio if args.defense == "advtrain" else None},
            "primary_scorer": T.PRIMARY_SCORER, "primary_mode": T.PRIMARY_MODE,
            "negative_set": "same attack payloads, clean (paired)",
        },
        "classes": classes,
        "verdict": verdict,
        "scorers": results,
        "tau_sweep": sweep,
        "complementarity": comp,
        "note": ("모드 A 는 캐스케이드가 이미 2차를 돌린 샘플만 판정하므로 추가 추론 비용이 "
                 "정확히 0 이다. 모드 B 는 전량 2차를 돌린 상한이라 비용이 융합형과 같다."),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"tamper_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fig_auc(results, FIG_DIR / f"tamper_auc_{tag}.png",
            f"M1 tampering detection: representation disagreement ({args.track})")
    fig_coverage(sweep, last["coverage_positive"], last["roc_auc"],
                 FIG_DIR / f"tamper_coverage_{tag}.png",
                 f"M1 mode-A coverage vs discriminability (tau sweep, k={args.budget})")
    print(f"\n[저장] 결과 → experiments/results/{out_json.name}")
    print(f"[저장] 그림 → docs/figures/attacks/tamper_auc_{tag}.png, tamper_coverage_{tag}.png")


if __name__ == "__main__":
    main()
