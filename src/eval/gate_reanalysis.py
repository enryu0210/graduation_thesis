"""학습 없는 재분석 E6·T4 (docs/paper_revision/03_prereg_v2.md) — seed 42 CV 확률을 다시 읽는다.

  E6: 과잉 넘김 배율 ρ 를 목표 완화(ε)와 분모 정의 둘로 다시 계산하고, 20% 감소를 검출할 검정력을 낸다.
      재분석이라 기존 판정(docs/14 §8.4·§8.6)을 바꾸지 않는다.
  T4: 운영점의 실사용자 층 정상 넘김률을 fold 도수를 합친 Wilson 구간으로 내고, S(π) 를 새 지연으로 다시 계산한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cross_validate
from src.eval import gate_confirm as GC
from src.eval import gate_frontier as GF
from src.eval import gate_g2
from src.eval.gate_cv import validate_alignment
from src.eval.metrics import wilson_ci
from src.models import gate_signals as gates

EPSILONS = (0.0, 0.001, 0.003, 0.005)      # 0 은 기존 판정과 같은 목표(대조용)
PREVALENCES = (0.001, 0.01, 0.033, 0.10, 0.50)
OUTPUT = GF.RESULTS / "gate_reanalysis_srbh_4class.json"


def power_for_reduction(b0, g2, reduction=0.20, alpha=0.05, test_train_ratio=0.25):
    """B0 평균의 reduction 만큼 줄었을 때 보정 t 검정(양측)이 그것을 잡아낼 확률. fold 간 분산은 관측값을 쓴다."""
    b0, g2 = np.asarray(b0, dtype=float), np.asarray(g2, dtype=float)
    spread = (b0 - g2).var(ddof=1)
    if len(b0) < 2 or spread == 0:
        return None
    df = len(b0) - 1
    noncentrality = reduction * b0.mean() / np.sqrt((1 / len(b0) + test_train_ratio) * spread)
    critical = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(critical, df, noncentrality) + stats.nct.cdf(-critical, df, noncentrality))


def over_deferral(reports, denominators):
    """ε·분모 정의별 과잉 넘김 배율. 목표에 닿지 못한 fold 가 있으면 그 칸은 검정하지 않는다."""
    table = {}
    for epsilon in EPSILONS:
        needed = {gate: [min((p["test_escalation_rate"] for p in row["curves"][gate]
                              if p["test_macro_f1"] >= row["stage2_test_macro_f1"] - epsilon), default=None)
                         for row in reports] for gate in ("b0", "g2")}
        for name, denominator in denominators.items():
            cell = {"needed_escalation": needed}
            if None in needed["b0"] or None in needed["g2"]:
                cell["reason"] = "목표에 닿지 못한 fold 가 있어 배율을 내지 않는다."
            else:
                b0, g2 = (np.array(needed[gate]) / denominator for gate in ("b0", "g2"))
                test = GC.corrected_t(b0 - g2)
                cell.update(b0_ratio=float(b0.mean()), g2_ratio=float(g2.mean()),
                            reduction=float(1 - g2.mean() / b0.mean()), corrected_p=test["p"],
                            b0_per_fold=b0.tolist(), g2_per_fold=g2.tolist(),
                            power_20pct=power_for_reduction(b0, g2))
            table[f"eps_{epsilon * 100:g}pp | {name}"] = cell
    return table


def operating_points(first, second, reports, strata):
    """val 에서 고른 운영점(2차 단독 val F1 의 0.5%p 이내, val 넘김 최소)의 층별 넘김 도수를 fold 에 걸쳐 합친다."""
    normal = first["classes"].tolist().index("Normal")
    counts = {gate: np.zeros(6, dtype=int) for gate in ("b0", "g2")}   # 실사용자 넘김·전체, 정상 넘김·전체, 공격 넘김·전체
    for row in reports:
        k = row["fold"]
        y, p1 = first["y"][first[f"test_idx_{k}"]], first[f"test_probs_{k}"]
        conf = p1.max(1)
        is_normal = y == normal
        is_real = is_normal & (strata[first[f"test_idx_{k}"]] == "real_user")
        for gate in counts:
            eligible = [p for p in row["curves"][gate] if p["val_macro_f1"] >= row["stage2_val_macro_f1"] - 0.005]
            if not eligible:
                raise ValueError(f"fold {k} {gate}: val 목표에 닿는 운영점이 없습니다.")
            chosen = min(eligible, key=lambda p: p["val_escalation_rate"])
            mask = gates.apply_classwise(p1, conf, chosen["taus"]) if "taus" in chosen else conf < chosen["tau"]
            counts[gate] += [mask[is_real].sum(), is_real.sum(), mask[is_normal].sum(), is_normal.sum(),
                             mask[~is_normal].sum(), (~is_normal).sum()]
    return counts


def speed_by_prevalence(counts, conditions):
    out = {}
    for gate, (real_k, real_n, normal_k, normal_n, attack_k, attack_n) in counts.items():
        r_attack = attack_k / attack_n
        row = {"real_user_r_N": wilson_ci(int(real_k), int(real_n)), "n_real_user": int(real_n),
               "val_mix_r_N": wilson_ci(int(normal_k), int(normal_n)), "r_A": float(r_attack), "speedup": {}}
        for name, (ms1, ms2) in conditions.items():
            for view, r_normal in (("val_mix", normal_k / normal_n), ("real_user", real_k / real_n)):
                row["speedup"][f"{name} | {view}"] = {
                    f"pi_{pi:g}": float(ms2 / (ms1 + (pi * r_attack + (1 - pi) * r_normal) * ms2))
                    for pi in PREVALENCES}
        out[gate] = row
    return out


def main():
    try:
        first, second, folds = validate_alignment(GF.STAGE1, GF.STAGE2)
        strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
        reports = gate_g2.fold_curves(first, second, folds, GF.STEP)
        wrong1 = {k: first[f"test_probs_{k}"].argmax(1) != first["y"][first[f"test_idx_{k}"]] for k in folds}
        right2 = {k: second[f"test_probs_{k}"].argmax(1) == first["y"][first[f"test_idx_{k}"]] for k in folds}
        denominators = {"e_o1 = stage-1 errors": np.array([wrong1[k].mean() for k in folds]),
                        "e_o2 = stage-1 wrong and stage-2 right": np.array([(wrong1[k] & right2[k]).mean() for k in folds])}
        result = {"note": "재분석(seed 42 CV) — 기존 판정을 바꾸지 않는다", "prereg": "docs/paper_revision/03_prereg_v2.md",
                  "e6_denominators": {name: values.tolist() for name, values in denominators.items()},
                  "e6": over_deferral(reports, denominators),
                  "t4": speed_by_prevalence(operating_points(first, second, reports, strata), GC.latency_conditions())}
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] 재분석 실패: {error}", file=sys.stderr)
        return 1
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, cell in result["e6"].items():
        if "reason" in cell:
            print(f"E6 | {name}: {cell['reason']}")
        else:
            print(f"E6 | {name}: B0 {cell['b0_ratio']:.2f} → G2 {cell['g2_ratio']:.2f} "
                  f"(감소 {cell['reduction'] * 100:+.1f}%, p={cell['corrected_p']}, 검정력 {cell['power_20pct']})")
    for gate, row in result["t4"].items():
        print(f"T4 | {gate}: 실사용자 r_N {row['real_user_r_N']} (n={row['n_real_user']}), r_A {row['r_A']:.4f}")
    print(f"저장: {OUTPUT.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
