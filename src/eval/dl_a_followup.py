"""DL-A 후속 — E13(G2 확인)·E14(k3 출구 → DistilBERT) 판정과 fold 4 진단(A2, 기술).

판정 규칙은 `docs/paper_revision/03_prereg_v2.md` §E13·§E14 를 그대로 옮긴 것이다(측정 전 커밋 d9d3678).
지연은 E12 실행(`dl_a_latency_srbh_4class_tok512-128.json`)의 t2·t_front·t_R(512)만 쓴다.

  --e13 --seed 44|45 : char-CNN → DistilBERT 에서 G2 vs B0 (실사용자, 합산 배수 이득·F1 손실·E_π 차 보정 t)
  --e14 --seed 42|43 : 1차 = charcnn_ee 보조 헤드(k3 출구) → DistilBERT 가 DL-A 보다 위인가
  --fold-diag        : seed 42 DL-A B0 운영점의 fold 별 넘김 분해(클래스·층·val τ) — 판정 아님
  --e16              : DL-C 보조 헤드 − k3 단독(인내 10)의 fold 평균 차 D (seed 42, 학습 안정성 대조)
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
from src.eval import cross_validate  # noqa: E402
from src.eval import dl_a  # noqa: E402
from src.eval import dl_cascade as DC  # noqa: E402
from src.eval import gate_confirm as GC  # noqa: E402
from src.eval import gate_frontier as GF  # noqa: E402
from src.eval.metrics import wilson_ci  # noqa: E402
from src.models import gate_signals as gates  # noqa: E402

MIN_GAIN = 0.20      # §E13 조건 1
MAX_LOSS_PP = 0.3    # §E13 조건 2 · §E14 조건 2
E14_RATIO = 1.2      # §E14 조건 1
E16_LIMIT_PP = 0.5   # §E16: E9 의 +1.01%p 의 절반
K3_PAT10 = "cvprobs_srbh_4class_charcnn_k3_raw_pat10_sealed.npz"
REAL = dict(GC.VIEWS)["real_user"]


def e12_latency() -> dict:
    path = dl_a.latency_path(dl_a.E12_TOKENS)
    return json.loads(path.read_text(encoding="utf-8"))["latency"]["ms"]


def strata() -> np.ndarray:
    return np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))


def operating(first: dict, second: dict, t1: float, t_r: float) -> dict:
    """실사용자 판본 secondary_rule(E10-3 과 같은 운영점 규칙). t1 = 1차 비용(t2 또는 t_front)."""
    paths, _ = GF.fold_paths(first, second, strata())
    row = GC.secondary_rule(paths, dl_a.fold_val_f1(second), REAL, t1, t_r)
    if "reason" in row:
        raise ValueError(row["reason"])
    row.pop("adopted")   # C-G2 채택 규칙이라 여기 판정과 무관하다.
    row["pooled_speedup"] = {g: dl_a.pooled_speedup(s) for g, s in row["per_fold_speedup"].items()}
    # S = t_R / (t1 + E·t_R) → fold 운영점 E_π 를 되돌린다(조건 3 의 대응 비교용).
    row["per_fold_e_pi"] = {g: [(t_r / s - t1) / t_r for s in v] for g, v in row["per_fold_speedup"].items()}
    row["paths"] = paths
    return row


def e13(seed: int) -> dict:
    ms = e12_latency()
    t2, t_r = ms["t2"], ms["t_R"]
    solo, bert = DC.load_cvprobs("charcnn", seed=seed), dl_a.load_npz(dl_a.distilbert_cvprobs(seed))
    DC.check_aligned(solo, bert)
    row = operating(solo, bert, t2, t_r)
    row.pop("paths")
    f1 = {g: float(np.mean(v)) for g, v in row["per_fold_macro_f1"].items()}
    e_diff = np.subtract(row["per_fold_e_pi"]["b0"], row["per_fold_e_pi"]["g2"])
    test = GC.corrected_t(e_diff)
    gain = row["pooled_speedup"]["g2"] / row["pooled_speedup"]["b0"] - 1
    checks = {"gain_ge_20pct": bool(gain >= MIN_GAIN),
              "f1_loss_le_0.3pp": bool(f1["g2"] >= f1["b0"] - MAX_LOSS_PP / 100),
              "e_pi_diff_significant": bool(e_diff.mean() > 0 and test["p"] is not None and test["p"] < 0.05)}
    f1_r = float(np.mean(DC.fold_test_f1(bert)))
    return {"prereg": "docs/paper_revision/03_prereg_v2.md §E13", "seed": seed, "latency_ms": ms,
            "distilbert_macro_f1": f1_r, "charcnn_macro_f1": float(np.mean(DC.fold_test_f1(solo))),
            "operating": row, "pooled_gain": gain, "f1_loss_pp": (f1["b0"] - f1["g2"]) * 100,
            "e_pi_diff_mean": float(e_diff.mean()), "e_pi_diff_corrected": test, **checks,
            "all_met": all(checks.values()),
            # 병기: 게이트별 E11 세 조건(기준 R = 같은 seed DistilBERT 단독).
            "e11_vs_r": {g: {"pooled_speedup_ge_1.2": bool(row["pooled_speedup"][g] >= DC.MIN_SPEEDUP),
                             "all_folds_gt_1": bool(min(row["per_fold_speedup"][g]) > 1),
                             "f1_vs_r_pp": (f1[g] - f1_r) * 100,
                             "f1_loss_le_0.3pp": bool(f1[g] >= f1_r - DC.MAX_F1_LOSS)} for g in ("b0", "g2")}}


def e14(seed: int) -> dict:
    ms = e12_latency()
    t2, t_front, t_r = ms["t2"], ms["t_front"], ms["t_R"]
    solo, bert = DC.load_cvprobs("charcnn", seed=seed), dl_a.load_npz(dl_a.distilbert_cvprobs(seed))
    aux = DC.load_cvprobs("charcnn_ee", aux=True, seed=seed)
    DC.check_aligned(solo, bert, aux)
    k3 = operating(aux, bert, t_front, t_r)
    base = operating(solo, bert, t2, t_r)
    for row in (k3, base):
        row.pop("paths")
    out = {"prereg": "docs/paper_revision/03_prereg_v2.md §E14", "seed": seed, "latency_ms": ms,
           "distilbert_macro_f1": float(np.mean(DC.fold_test_f1(bert))),
           "aux_solo_macro_f1": float(np.mean(DC.fold_test_f1(aux))), "e14": k3, "dl_a": base}
    for gate in ("b0", "g2"):
        ratio = k3["pooled_speedup"][gate] / base["pooled_speedup"][gate]
        f1_k3 = float(np.mean(k3["per_fold_macro_f1"][gate]))
        f1_base = float(np.mean(base["per_fold_macro_f1"][gate]))
        checks = {"ratio_ge_1.2": bool(ratio >= E14_RATIO),
                  "f1_loss_le_0.3pp": bool(f1_k3 >= f1_base - MAX_LOSS_PP / 100)}
        # 판정은 B0 만(§E14). G2 는 같은 계산의 병기.
        out[f"vs_dl_a_{gate}"] = {"speedup_ratio": ratio, "f1_diff_pp": (f1_k3 - f1_base) * 100,
                                  **checks, "above": all(checks.values()),
                                  "f1_vs_dl_a_corrected": GC.corrected_t(
                                      np.subtract(k3["per_fold_macro_f1"][gate], base["per_fold_macro_f1"][gate]))}
    return out


def best_epochs(tag: str) -> list[int]:
    data = json.loads((GF.RESULTS / f"cv_{tag}.json").read_text(encoding="utf-8"))
    return [int(f["best_epoch"]) for f in data["fold_results"]]


def e16() -> dict:
    """§E16: 인내만 5 → 10 으로 바꾼 k3 단독과 DL-C 보조 헤드(seed 42)의 fold 대응 차."""
    aux = DC.load_cvprobs("charcnn_ee", aux=True)
    k3 = dl_a.load_npz(GF.RESULTS / K3_PAT10)
    k3_old = DC.load_cvprobs("charcnn_k3")
    DC.check_aligned(aux, k3, k3_old)
    f1 = {"aux": DC.fold_test_f1(aux), "k3_pat10": DC.fold_test_f1(k3), "k3_pat5": DC.fold_test_f1(k3_old)}
    diffs = np.subtract(f1["aux"], f1["k3_pat10"])
    d_pp = float(diffs.mean() * 100)
    return {"prereg": "docs/paper_revision/03_prereg_v2.md §E16", "seed": DC.BASE_SEED,
            "per_fold_macro_f1": f1, "macro_f1": {k: float(np.mean(v)) for k, v in f1.items()},
            "D_pp": d_pp, "per_fold_D_pp": (diffs * 100).tolist(), "D_corrected": GC.corrected_t(diffs),
            # 병기: 인내 5 → 10 이 k3 단독 자체를 얼마나 올렸나.
            "pat10_minus_pat5_corrected": GC.corrected_t(np.subtract(f1["k3_pat10"], f1["k3_pat5"])),
            "best_epoch": {"aux": best_epochs("srbh_4class_charcnn_ee_raw_sealed"),
                           "k3_pat10": best_epochs("srbh_4class_charcnn_k3_raw_pat10_sealed"),
                           "k3_pat5": best_epochs("srbh_4class_charcnn_k3_raw_sealed")},
            "verdict": "학습 안정성으로 상당 부분 설명" if d_pp <= E16_LIMIT_PP else "인내를 늘려도 차이가 남는다"}


def fold_diag(seed: int = DC.BASE_SEED) -> dict:
    """A2: B0 운영점(E10 지연 — E10-3 과 같은 운영점)에서 fold 별 넘김을 분해한다."""
    ms = json.loads(dl_a.latency_path((dl_a.FULL_TOKENS,)).read_text(encoding="utf-8"))["latency"]["ms"]
    solo, bert = DC.load_cvprobs("charcnn", seed=seed), dl_a.load_npz(dl_a.distilbert_cvprobs(seed))
    DC.check_aligned(solo, bert)
    row = operating(solo, bert, ms["t2"], ms["t_R"])
    layer = strata()
    classes = solo["classes"].tolist()
    normal = classes.index("Normal")
    folds = []
    for k in range(1, 6):
        conf_val = solo[f"val_probs_{k}"].max(1)
        tau = GF.b0_taus(conf_val)[row["per_fold_index"]["b0"][k - 1]]
        rows = {}
        for split in ("val", "test"):
            idx = solo[f"{split}_idx_{k}"]
            y, p = solo["y"][idx], solo[f"{split}_probs_{k}"]
            deferred = p.max(1) < tau
            real = (y == normal) & (layer[idx] == "real_user")
            scanner = (y == normal) & (layer[idx] != "real_user")
            n_real = int(real.sum())
            rows[split] = {
                "r_N_real": wilson_ci(int(deferred[real].sum()), n_real) | {"n": n_real},
                "r_N_scanner": float(deferred[scanner].mean()) if scanner.any() else None,
                "r_A": float(deferred[y != normal].mean()),
                "by_true_class": {c: float(deferred[y == i].mean()) for i, c in enumerate(classes)},
                # 실사용자 정상 중 넘긴 요청이 1차에서 무엇으로 예측됐나 — 오탐성 넘김인지 본다.
                "real_deferred_pred": {c: int((deferred & real & (p.argmax(1) == i)).sum())
                                       for i, c in enumerate(classes)},
                "real_conf_median": float(np.median(p.max(1)[real])) if n_real else None,
            }
        folds.append({"fold": k, "tau": float(tau), "budget_index": int(row["per_fold_index"]["b0"][k - 1]),
                      "speedup": row["per_fold_speedup"]["b0"][k - 1], **rows})
    return {"kind": "기술(판정 아님) — 03_prereg_v2.md 기술 절 A2", "seed": seed, "latency_ms": ms, "folds": folds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--e13", action="store_true")
    mode.add_argument("--e14", action="store_true")
    mode.add_argument("--fold-diag", action="store_true")
    mode.add_argument("--e16", action="store_true")
    parser.add_argument("--seed", type=int, default=DC.BASE_SEED)
    args = parser.parse_args()
    if args.e13 and args.seed in dl_a.E10_E11_SEEDS:
        parser.error("E13 은 새 seed(44·45)로만 판정한다 — 42·43 은 이미 본 seed 다(§E13).")
    try:
        if args.e13:
            result, name = e13(args.seed), f"dl_a_e13_srbh_4class_s{args.seed}.json"
        elif args.e16:
            result, name = e16(), "dl_a_e16_srbh_4class.json"
        elif args.e14:
            result, name = e14(args.seed), f"dl_a_e14_srbh_4class{DC.cvprobs_suffix(args.seed)}.json"
        else:
            result, name = fold_diag(args.seed), f"dl_a_fold_diag_srbh_4class{DC.cvprobs_suffix(args.seed)}.json"
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] {error}", file=sys.stderr)
        return 1
    out = GF.RESULTS / name
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("operating", "e14", "dl_a", "folds")},
                     ensure_ascii=False, indent=1, default=float))
    if "folds" in result:
        for f in result["folds"]:
            print(f"fold {f['fold']}: τ {f['tau']:.4f} S {f['speedup']:.2f} | test r_N_real {f['test']['r_N_real']}"
                  f" r_A {f['test']['r_A']:.4f} | val r_N_real {f['val']['r_N_real']}")
    print(f"저장: {out.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
