"""정확도–속도 경계 비교(docs/14 §8.20, 탐색) — B0(msp 단일 τ) vs G2(클래스별 τ) 를 곡선 전체로 비교한다.

교수 기준(2026-10-07): "개선"은 비교 대상보다 정확도(Y)–속도(X) 그래프에서 위에 있으면 쓸 수 있다.
사전 등록 판정(§8.5 과잉 넘김 배율)은 곡선 꼬리 한 점만 봤으므로, 같은 CV 확률로 곡선 전체를 다시 본다.
⚠️ G2 를 찾아낸 CV 데이터를 그대로 쓰므로 판정이 아니라 탐색이다 — "개선" 확정은 새 seed 확인 실험으로.

X = 배포 속도 배수 S = t2 / (t1 + E_π·t2), E_π = π·r_A + (1−π)·r_N (π=3.3%).
r_N 은 두 판본: fold-test 정상 전체(96% 스캐너) / 실사용자 층만(G4 보정).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cross_validate
from src.models import gate_signals as gates

RESULTS = PROJECT_ROOT / "experiments/results"
STAGE1 = RESULTS / "cvprobs_srbh_4class_cnn_raw_rgb_sealed.npz"
STAGE2 = RESULTS / "cvprobs_srbh_4class_charcnn_raw_sealed.npz"
FIGURE = PROJECT_ROOT / "docs/figures/cascade/gate_frontier_b0_g2_srbh_4class.png"
OUTPUT = RESULTS / "gate_frontier_srbh_4class.json"
# T2(docs/14 §8.3) 실측 ms/요청. ⚠️ GPU·128 의 char-CNN 값은 배치 32·1024 보다 느린 단일 측정(재측정 필요).
DEVICES = {"CPU, batch 1": (0.350, 1.114), "GPU, batch 128": (0.0040, 0.0699)}
PI = 0.033
STEP = 0.005
NORMAL_VIEWS = (("val_mix", "Normal = val mix (96% scanner)"), ("real_user", "Normal = real users (G4-corrected)"))


def speedup(r_normal, r_attack, ms1, ms2):
    return ms2 / (ms1 + (PI * r_attack + (1 - PI) * r_normal) * ms2)


def frontier(speeds, f1s, grid):
    """각 속도 이상에서 낼 수 있는 최고 F1(파레토 경계). 그 속도에 닿는 점이 없으면 NaN.
    게이트 경로가 예산 순서로 단조롭지 않아도(G2 탐욕 경로) 경계로 비교해야 공정하다."""
    speeds, f1s = np.asarray(speeds), np.asarray(f1s)
    return np.array([f1s[speeds >= x].max() if (speeds >= x).any() else np.nan for x in grid])


def fold_paths(first, second, strata):
    """fold 마다 게이트 경로의 점별 (r_N 전체, r_N 실사용자, r_A, test F1) 과 단독 모델 F1 을 모은다."""
    classes = first["classes"].tolist()
    normal, n_classes = classes.index("Normal"), len(classes)
    paths, solo = {"b0": [], "g2": []}, []
    for k in range(1, 6):
        y_val, y_test = first["y"][first[f"val_idx_{k}"]], first["y"][first[f"test_idx_{k}"]]
        p1v, p1t = first[f"val_probs_{k}"], first[f"test_probs_{k}"]
        p2v, p2t = second[f"val_probs_{k}"], second[f"test_probs_{k}"]
        conf_val, conf_test = p1v.max(1), p1t.max(1)
        is_normal, is_attack = y_test == normal, y_test != normal
        is_real = is_normal & (strata[first[f"test_idx_{k}"]] == "real_user")
        masks = {"b0": [conf_test < gates.tau_for_budget(conf_val, q) for q in np.r_[np.arange(0, 1, STEP), 1.0]],
                 "g2": [gates.apply_classwise(p1t, conf_test, point["taus"])
                        for point in gates.classwise_tau_path(p1v, p2v, y_val, n_classes, STEP)]}
        for gate, gate_masks in masks.items():
            paths[gate].append(np.array([
                (m[is_normal].mean(), m[is_real].mean(), m[is_attack].mean(),
                 gates._macro_f1_fast(y_test, np.where(m, p2t.argmax(1), p1t.argmax(1)), n_classes))
                for m in gate_masks]))
        solo.append((gates._macro_f1_fast(y_test, p1t.argmax(1), n_classes),
                     gates._macro_f1_fast(y_test, p2t.argmax(1), n_classes)))
    return paths, np.array(solo).mean(0)


def envelopes(paths, ms1, ms2, view_col, grid):
    """게이트별 fold × grid 경계 행렬."""
    return {gate: np.array([frontier(speedup(P[:, view_col], P[:, 2], ms1, ms2), P[:, 3], grid) for P in fold_list])
            for gate, fold_list in paths.items()}


def summarize_gap(env, checkpoints):
    """속도 지점별 G2−B0 F1 차이(pp)와 G2 가 위인 fold 수."""
    rows = {}
    for x, column in checkpoints:
        diff = env["g2"][:, column] - env["b0"][:, column]
        if not np.isnan(diff).any():
            rows[f"{x:.2f}"] = {"mean_pp": float(diff.mean() * 100), "min_pp": float(diff.min() * 100),
                                "folds_above": int((diff > 0).sum())}
    return rows


def plot(paths, solo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    summary = {}
    for row, (device, (ms1, ms2)) in enumerate(DEVICES.items()):
        grid = np.linspace(1.0, ms2 / ms1, 200)
        for col, (view, title) in enumerate(NORMAL_VIEWS):
            env = envelopes(paths, ms1, ms2, col, grid)
            ax = axes[row, col]
            for gate, color, name in (("b0", "tab:blue", "B0: single tau on MSP (standard)"),
                                      ("g2", "tab:red", "G2: class-wise tau")):
                # 경계 바깥(NaN 만 있는 속도 지점)의 nanmean 경고는 의도된 빈칸이라 숨긴다.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ax.plot(grid, np.nanmean(env[gate], 0), color=color, lw=2, label=name)
                    ax.fill_between(grid, np.nanmin(env[gate], 0), np.nanmax(env[gate], 0), color=color, alpha=.15)
            ax.scatter([1.0], [solo[1]], marker="*", s=180, c="k", zorder=5, label=f"char-CNN alone (F1 {solo[1]:.3f})")
            ax.scatter([ms2 / ms1], [solo[0]], marker="s", s=70, c="gray", zorder=5,
                       label=f"RGB CNN alone (F1 {solo[0]:.3f})")
            ax.set_title(f"{device} | {title}", fontsize=10)
            ax.set_xlabel("Speedup vs char-CNN alone (attack share 3.3%)")
            ax.set_ylim(0.88, 0.995)
            ax.grid(alpha=.3)
            if col == 0:
                ax.set_ylabel("Test Macro-F1 (5-fold CV, band = fold min-max)")
            checkpoints = [(x, int(np.argmin(abs(grid - x)))) for x in np.arange(1.0, ms2 / ms1, 0.25 if ms2 / ms1 < 5 else 1.0)]
            summary[f"{device} | {view}"] = summarize_gap(env, checkpoints)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Cascade accuracy-speed frontier: B0 vs G2 (srbh_4class, exploratory)", fontsize=12)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=130)
    return summary


def oracle_floor(first, second, strata):
    """정상 넘김률의 하한: 1차가 틀리고 2차가 맞는 정상 행만 넘기는 오라클(fold-test 풀)."""
    normal = first["classes"].tolist().index("Normal")
    idx = np.concatenate([first[f"test_idx_{k}"] for k in range(1, 6)])
    p1 = np.concatenate([first[f"test_probs_{k}"] for k in range(1, 6)])
    p2 = np.concatenate([second[f"test_probs_{k}"] for k in range(1, 6)])
    y, layer = first["y"][idx], strata[idx]
    fixable = (p1.argmax(1) != y) & (p2.argmax(1) == y)
    is_normal = y == normal
    return {"r_N_all": float(fixable[is_normal].mean()),
            "r_N_real_user": float(fixable[is_normal & (layer == "real_user")].mean()),
            "stage1_normal_error": float((p1.argmax(1) != y)[is_normal].mean())}


def main():
    try:
        with np.load(STAGE1, allow_pickle=False) as a, np.load(STAGE2, allow_pickle=False) as b:
            first, second = dict(a), dict(b)
        if str(first["label_fingerprint"]) != str(second["label_fingerprint"]):
            raise ValueError("두 cvprobs 의 라벨 지문이 다릅니다.")
        strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
        if len(strata) != len(first["y"]):
            raise ValueError("층 배열 길이가 CV 풀과 다릅니다.")
        paths, solo = fold_paths(first, second, strata)
        result = {"note": "탐색(G2 발견에 쓴 CV 데이터) — 판정 아님", "pi": PI, "devices_ms": DEVICES,
                  "solo_macro_f1": {"rgb_cnn": float(solo[0]), "charcnn": float(solo[1])},
                  "oracle": oracle_floor(first, second, strata), "g2_minus_b0": plot(paths, solo)}
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"그림: {FIGURE.relative_to(PROJECT_ROOT)}")
    except (OSError, ValueError, KeyError) as error:
        print(f"경계 비교 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
