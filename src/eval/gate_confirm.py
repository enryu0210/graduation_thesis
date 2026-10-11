"""C-G2 확인 실험 판정 — 새 seed 의 CV 확률로 B0(msp 단일 τ)와 G2(클래스별 τ)의 경계를 다시 비교한다.

판정 규칙은 `docs/paper_revision/03_prereg_v2.md` 의 C-G2 를 그대로 옮긴 것이다(측정 전 확정, 커밋 2a456d7).
방법은 gate_frontier.py 와 같고 입력 seed 만 바꾼다. seed 42 로 돌린 값은 G2 를 찾아낸 자료라 탐색이다.

  1차 규칙(경계 비교): 배포 넘김률 E_π 네 지점에서 "그 넘김 이하의 최고 fold-test Macro-F1" 을 비교한다.
  2차 규칙(E7): 운영점을 val 에서 하나 고른 뒤(편향 없는 판본) fold-test 배포 배수 S 를 비교한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cross_validate
from src.eval import gate_frontier as GF
from src.models import gate_signals as gates

# 판정 지점은 E_π 로 고정한다 — 지연이 바뀌어도 움직이지 않는다(기존 지연의 1.25·1.50·1.75·2.00배와 같은 뜻).
E_POINTS = (0.486, 0.353, 0.257, 0.186)
LEGACY_CPU = ("CPU b1, forward only, padded 2304 (old T2)", (0.350, 1.114))
LATENCY_JSON = GF.RESULTS / "latency_fair_srbh_4class.json"
# fold_paths 의 열 번호.
R_ALL, R_REAL, R_ATTACK, F1_TEST, VAL_R_N, VAL_R_A, VAL_F1 = range(7)
VIEWS = (("val_mix", R_ALL), ("real_user", R_REAL))


def corrected_t(diffs, test_train_ratio=0.25):
    """Nadeau–Bengio 보정 t. fold 의 학습 집합이 겹쳐 분산이 과소추정되는 것을 (1/k + n_test/n_train) 으로 키운다."""
    d = np.asarray(diffs, dtype=float)
    variance = d.var(ddof=1)
    if len(d) < 2 or variance == 0:
        return {"t": None, "p": None}
    t = d.mean() / np.sqrt((1 / len(d) + test_train_ratio) * variance)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), len(d) - 1))}


def e_pi(path, normal_col, attack_col=R_ATTACK):
    return GF.PI * path[:, attack_col] + (1 - GF.PI) * path[:, normal_col]


def primary_rule(paths, normal_col):
    """네 E_π 지점의 경계 비교. 넘김 0% 점이 항상 경로에 있어 빈 지점은 생기지 않는다."""
    best = {gate: np.array([[P[e_pi(P, normal_col) <= e, F1_TEST].max() for e in E_POINTS] for P in folds])
            for gate, folds in paths.items()}
    diff = best["g2"] - best["b0"]                      # fold × 지점
    folds_above = (diff > 0).sum(0)
    rule_i = bool((diff.mean(0) > 0).all())
    rule_ii = bool((folds_above >= 4).sum() >= 3)
    test = corrected_t(diff.mean(1))
    return {"points": {f"{e:.3f}": {"b0_f1": float(best["b0"][:, j].mean()), "g2_f1": float(best["g2"][:, j].mean()),
                                    "mean_pp": float(diff[:, j].mean() * 100), "min_pp": float(diff[:, j].min() * 100),
                                    "folds_above": int(folds_above[j])} for j, e in enumerate(E_POINTS)},
            "rule_i_all_points_positive": rule_i, "rule_ii_folds": rule_ii, "passed": rule_i and rule_ii,
            "fold_mean_diff_pp": (diff.mean(1) * 100).tolist(), **{f"corrected_{k}": v for k, v in test.items()},
            "significant": test["p"] is not None and test["p"] < 0.05}


def secondary_rule(paths, val_solo2, normal_col, ms1, ms2):
    """E7: val Macro-F1 이 char-CNN 단독의 0.5%p 이내인 점 중 val E_π 최소(= S 최대)를 게이트별로 고른다."""
    chosen = {}
    for gate, folds in paths.items():
        rows = []
        for P, target in zip(folds, val_solo2):
            eligible = np.flatnonzero(P[:, VAL_F1] >= target - 0.005)
            if not len(eligible):
                return {"reason": f"{gate}: val 목표에 닿는 점이 없는 fold 가 있어 판정하지 않는다."}
            i = eligible[np.argmin(e_pi(P[eligible], VAL_R_N, VAL_R_A))]
            rows.append((ms2 / (ms1 + e_pi(P[i:i + 1], normal_col)[0] * ms2), P[i, F1_TEST],
                         e_pi(P[i:i + 1], normal_col)[0], P[i, normal_col], P[i, R_ATTACK]))
        chosen[gate] = np.array(rows)
    s_b0, s_g2 = chosen["b0"][:, 0], chosen["g2"][:, 0]
    gain = float(s_g2.mean() / s_b0.mean() - 1)
    loss_pp = float((chosen["b0"][:, 1] - chosen["g2"][:, 1]).mean() * 100)
    test = corrected_t(s_g2 - s_b0)
    out = {gate: dict(zip(("speedup", "macro_f1", "e_pi", "r_N", "r_A"), rows.mean(0).tolist()))
           for gate, rows in chosen.items()}
    out.update(speedup_gain=gain, f1_loss_vs_b0_pp=loss_pp, corrected_t=test["t"], corrected_p=test["p"],
               per_fold_speedup={gate: rows[:, 0].tolist() for gate, rows in chosen.items()},
               adopted=bool(gain >= 0.20 and test["p"] is not None and test["p"] < 0.05 and loss_pp <= 0.3))
    return out


def latency_conditions():
    """기존 지연과 E1·E1-G 재측정 지연. 재측정 파일이 없으면 기존 지연만 쓴다."""
    conditions = dict([LEGACY_CPU])
    if LATENCY_JSON.is_file():
        fair = json.loads(LATENCY_JSON.read_text(encoding="utf-8"))
        for row in fair["cpu"]:
            c = row["components"]
            conditions[f"CPU b1, {row['threads']} thread, end-to-end, real length (E1)"] = (
                c["stage1_e2e"]["p50"], c["stage2_real_e2e"]["p50"])
        for row in (fair.get("gpu") or {}).get("rows", []):
            conditions[f"GPU b{row['batch']}, forward only, batch-max padding (E1-G)"] = (
                row["stage1_ms"], row["stage2_batchmax_ms"])
            conditions[f"GPU b{row['batch']}, forward only, padded 2304 (E1-G)"] = (
                row["stage1_ms"], row["stage2_pad2304_ms"])
    return conditions


def analyze(seed, figure_devices=None):
    suffix = "" if seed == 42 else f"_s{seed}"
    stage1 = GF.RESULTS / f"cvprobs_srbh_4class_cnn_raw_rgb_sealed{suffix}.npz"
    stage2 = GF.RESULTS / f"cvprobs_srbh_4class_charcnn_raw_sealed{suffix}.npz"
    with np.load(stage1, allow_pickle=False) as a, np.load(stage2, allow_pickle=False) as b:
        first, second = dict(a), dict(b)
    if str(first["label_fingerprint"]) != str(second["label_fingerprint"]):
        raise ValueError("두 cvprobs 의 라벨 지문이 다릅니다.")
    for k in range(1, 6):   # 두 모델이 같은 fold·같은 내부 val 을 봤는지 — 어긋나면 넘김 비교가 무의미하다.
        if not all(np.array_equal(first[f"{s}_idx_{k}"], second[f"{s}_idx_{k}"]) for s in ("val", "test")):
            raise ValueError(f"fold {k} 인덱스가 두 모델에서 다릅니다.")
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    paths, solo = GF.fold_paths(first, second, strata)
    n_classes = len(first["classes"])
    val_solo2 = [gates._macro_f1_fast(first["y"][first[f"val_idx_{k}"]], second[f"val_probs_{k}"].argmax(1), n_classes)
                 for k in range(1, 6)]
    conditions = latency_conditions()
    result = {"seed": seed, "status": "탐색(G2 발견 자료)" if seed == 42 else "확인(사전 기준 C-G2)",
              "prereg": "docs/paper_revision/03_prereg_v2.md", "pi": GF.PI, "e_points": E_POINTS,
              "solo_macro_f1": {"rgb_cnn": float(solo[0]), "charcnn": float(solo[1])},
              "primary": {view: primary_rule(paths, col) for view, col in VIEWS},
              "secondary_e7": {name: {"ms1": ms1, "ms2": ms2, "upper_bound": ms2 / ms1,
                                      **{view: secondary_rule(paths, val_solo2, col, ms1, ms2) for view, col in VIEWS}}
                               for name, (ms1, ms2) in conditions.items()}}
    if figure_devices:
        figure = GF.FIGURE.with_name(f"gate_frontier_b0_g2_srbh_4class_s{seed}.png")
        GF.plot(paths, solo, figure=figure, devices={name: conditions[name] for name in figure_devices},
                title=f"confirmation, CV seed {seed}")
        result["figure"] = figure.relative_to(PROJECT_ROOT).as_posix()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--figure-devices", nargs=2, metavar="NAME",
                        help="그림 두 줄에 쓸 지연 조건 이름(산출 JSON 의 secondary_e7 키)")
    args = parser.parse_args()
    try:
        result = analyze(args.seed, args.figure_devices)
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] C-G2 판정 실패: {error}", file=sys.stderr)
        return 1
    output = GF.RESULTS / f"gate_confirm_srbh_4class_s{args.seed}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("seed", "status", "solo_macro_f1", "primary")},
                     ensure_ascii=False, indent=2))
    for name, block in result["secondary_e7"].items():
        for view, _ in VIEWS:
            row = block[view]
            if "reason" in row:
                print(f"E7 | {name} | {view}: {row['reason']}")
                continue
            print(f"E7 | {name} | {view}: B0 {row['b0']['speedup']:.3f}배 → G2 {row['g2']['speedup']:.3f}배 "
                  f"({row['speedup_gain'] * 100:+.1f}%, p={row['corrected_p']}, F1 손실 {row['f1_loss_vs_b0_pp']:+.2f}pp) "
                  f"채택={row['adopted']}")
    print(f"저장: {output.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
