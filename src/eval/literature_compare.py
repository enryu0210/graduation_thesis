"""문헌 비교 그림 — 유사 연구가 보고한 최고 수치와 우리 결과를 정확도(Y)–속도(X) 평면에 함께 올린다.

왜 두 종류의 패널인가:
    (a) 논문에 적힌 값 그대로. 데이터·클래스 수·하드웨어가 서로 달라 **직접 비교는 안 된다** — 위치 감각만 준다.
    (b)(c) 같은 데이터·같은 기기에서 다시 잰 값. 유사 연구의 "값싼 선형 1차 → 깊은 2차" 구성을
        우리 2차(char-CNN)로 재구현해(linear_stage1.py, E3) 우리 캐스케이드와 같은 축에 놓는다.
        교수 기준("그래프에서 위면 개선")은 이 패널로만 판단할 수 있다.

선행 조건: latency_fair.py(E1) → linear_stage1.py --fit → --latency → --analyze.

--dl-only: 고전 ML(선형·부스팅)을 모두 빼고 딥러닝 모델만 남긴 판을 따로 그린다(파일명 `_dl_only`, 기본 그림은 그대로).
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
from src.eval import cross_validate as CV
from src.eval import gate_frontier as GF
from src.eval import linear_stage1 as LS

FIGURE = PROJECT_ROOT / "docs/figures/cascade/literature_comparison_srbh_4class.png"
# (범례 이름, 요청당 지연 ms, 점수, 표식, 색, 점수가 F1 인가) — 출처 표는 docs/12 §1.9, 해석은 docs/14 §8.26.
#   Tasdemir 등(arXiv:2312.13041) Table 5·6·식 14: SQL 주입 이진 30,609건, A100 GPU.
#   WAMM(arXiv:2512.23610) Table VI: SR-BH 2020 원본·정제본, 8 vCPU 서버, Macro-F1.
#   Kutlimuratov 등(Computers 15(3):174) Table 9: UNSW-NB15, 정확도(F1 미보고).
REPORTED = [
    ("Tasdemir'23 stage 1, PassiveAggressive (SQLi, GPU)", 0.0003, 0.9975, "v", "tab:green", True),
    ("Tasdemir'23 cascade (SQLi, GPU)", 0.0988, 0.9981, "D", "tab:green", True),
    ("Tasdemir'23 BERT alone (SQLi, GPU)", 2.0439, 0.9983, "^", "tab:green", True),
    ("WAMM'25 XGBoost (SR-BH original, CPU)", 0.071, 0.9677, "o", "tab:blue", True),
    ("WAMM'25 XGBoost (SR-BH deduplicated, CPU)", 0.061, 0.9705, "s", "tab:blue", True),
    ("Kutlimuratov'26 cascade (UNSW-NB15, accuracy)", 0.0036, 0.9481, "P", "tab:gray", False),
]
FIGURE_DL = FIGURE.with_name("literature_comparison_dl_only_srbh_4class.png")
# 딥러닝만 남긴 판의 보고값. 뺀 것: 고전 ML 단독(PassiveAggressive·XGBoost), 1차가 고전 ML 인 Tasdemir 캐스케이드,
#   모델 종류를 원문에서 확인하지 못한 Kutlimuratov 캐스케이드. WAMM 의 CNN-BiLSTM(0.7522)은 지연을 옮겨 적지 않아 뺐다.
REPORTED_DL = [
    ("Tasdemir'23 BERT alone (SQLi, GPU)", 2.0439, 0.9983, "^", "tab:green", True),
    ("WAMM'25 DistilBERT (SR-BH original, CPU)", 14.423, 0.8547, "o", "tab:blue", True),
    ("WAMM'25 BiLSTM (SR-BH original, CPU)", 0.713, 0.7711, "s", "tab:blue", True),
]
# 지연을 보고하지 않은 연구는 가로선으로만 표시한다(둘 다 SR-BH 2020 이진 탐지).
NO_LATENCY = [("WADBERT'26 (SR-BH, binary F1 0.995, no latency reported)", 0.9950, "tab:brown"),
              ("X-WAD'26 (SR-BH relabeled, binary F1 0.988, no latency reported)", 0.9884, "tab:olive")]
STAGE1_STYLE = {"rgb_cnn": ("tab:red", "RGB CNN (ours)"), "tfidflr": ("tab:purple", "TF-IDF + LogReg (re-impl.)"),
                "hashlr": ("tab:orange", "Hashing + LogReg (re-impl.)"),
                # E2 후보 가운데 지연이 절반인 것과 정확도가 가장 높은 것만 올린다(셋 다 그리면 선이 겹쳐 읽히지 않는다).
                "cnn_crop8": ("tab:brown", "RGB CNN top 8 rows (ours)"), "cnn_1d": ("tab:cyan", "RGB 1-D CNN (ours)")}
CLASSICAL_STAGE1 = {"tfidflr", "hashlr"}
# 딥러닝 판에만 올리는 char-CNN 내부 조기 종료(DL-C, E8 운영점)와 k3 단독 대조(E9-1). 기술만 한다(사전 문서 §E9-3).
DL_CASCADE = GF.RESULTS / "dl_cascade_srbh_4class.json"
# 같은 SR-BH fold 로 재구현한 WAMM DistilBERT(E10-2 의 R)와 char-CNN → DistilBERT 운영점(E10-3). 기술만 한다.
DL_A = GF.RESULTS / "dl_a_srbh_4class.json"
DL_A_E14 = GF.RESULTS / "dl_a_e14_srbh_4class.json"
DL_VIEWS = (("val_mix", "Validation benign mix (96% scanner)"),
            ("real_user", "Real-user benign traffic (browser UA)"))


def plot_dl_only(strata):
    """RQ1(RGB-CNN)을 제외하고 논문 본문(char-CNN · DistilBERT · 캐스케이드)만 남긴 깔끔한 비교 그림."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval import dl_a, dl_cascade as DC, gate_confirm as GC

    dla = json.loads(DL_A.read_text(encoding="utf-8"))
    dlc = json.loads(DL_CASCADE.read_text(encoding="utf-8"))
    e14 = json.loads(DL_A_E14.read_text(encoding="utf-8"))
    t2, t_r = dla["latency_ms"]["t2"], dla["latency_ms"]["t_R"]
    f1_cnn, f1_bert = dla["charcnn_macro_f1"], dla["distilbert_macro_f1"]

    solo = DC.load_cvprobs("charcnn", seed=42)
    bert = dl_a.load_npz(dl_a.distilbert_cvprobs(42))
    ee_aux = DC.load_cvprobs("charcnn_ee", aux=True, seed=42)
    paths_dla, _ = GF.fold_paths(solo, bert, strata)
    paths_e14, _ = GF.fold_paths(ee_aux, bert, strata)
    t_front_14, t_r_14 = e14["latency_ms"]["t_front"], e14["latency_ms"]["t_R"]
    e14_views = {
        "val_mix": GC.secondary_rule(paths_e14, dl_a.fold_val_f1(bert), 0, t_front_14, t_r_14),
        "real_user": e14["e14"],
    }
    # τ = 0 (넘김 0%, 1차 단독 끝점)을 경로에 명시해 모든 폴드 곡선이 1차 단독 점까지 끊기지 않고 이어지게 한다.
    for paths, first in ((paths_dla, solo), (paths_e14, ee_aux)):
        f1s = DC.fold_test_f1(first)
        for gate in ("b0", "g2"):
            paths[gate] = [np.vstack([np.array([[0.0, 0.0, 0.0, f1s[k], 0.0, 0.0, 0.0]]), p])
                           for k, p in enumerate(paths[gate])]

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    for label, ms, score, marker, color, is_f1 in REPORTED_DL:
        ax.scatter([ms], [score], marker=marker, s=90, zorder=4, label=label,
                   facecolors=color if is_f1 else "none", edgecolors=color)
    ours_a = [
        ("DistilBERT alone (ours, 512-tok)", t_r, f1_bert, "h", "tab:purple"),
        ("char-CNN -> DistilBERT cascade (ours)",
         t_r / dla["e10_3"]["real_user"]["pooled_speedup"]["b0"],
         dla["e10_3"]["real_user"]["b0"]["macro_f1"], "D", "tab:blue"),
        ("Early-exit -> DistilBERT cascade (ours)",
         t_r_14 / e14["e14"]["pooled_speedup"]["b0"],
         e14["e14"]["b0"]["macro_f1"], "D", "tab:green"),
        ("char-CNN alone (ours)", t2, f1_cnn, "*", "k"),
        ("char-CNN early exit (ours)",
         t2 / dlc["dl_c"]["real_user"]["b0"]["speedup"],
         dlc["dl_c"]["real_user"]["b0"]["macro_f1"], "s", "darkgoldenrod"),
    ]
    for label, ms, score, marker, color in ours_a:
        ax.scatter([ms], [score], marker=marker, s=170 if marker == "*" else 95, color=color, zorder=5,
                   edgecolors="k", linewidths=.8, label=label)
    no_lat_short = [
        ("WADBERT'26 (SR-BH binary, no latency)", 0.9950, "tab:brown"),
        ("X-WAD'26 (SR-BH binary, no latency)", 0.9884, "tab:olive"),
    ]
    for label, score, color in no_lat_short:
        ax.axhline(score, color=color, lw=1.1, ls="--", label=label)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0.74, 1.002)
    ax.set_xlabel("Latency per request, ms (log scale; faster to the right)")
    ax.set_ylabel("Macro-F1 as reported (dashed lines = binary F1)")
    ax.set_title("(a) As reported vs. evaluated (prior work uses different setups)", fontsize=10)
    ax.grid(alpha=.3, which="both")
    legends = [ax.legend(fontsize=8.2, loc="upper left", bbox_to_anchor=(-0.04, -0.14), ncol=2, frameon=False)]

    max_speed = t_r / t2
    grid = np.linspace(1.0, max_speed, 300)
    grid_14 = np.linspace(1.0, min(100.0, t_r_14 / t_front_14), 300)
    for ax, (col, (view_key, view_title)), letter in zip(axes[1:], enumerate(DL_VIEWS), "bc"):
        env_dla = GF.envelopes(paths_dla, t2, t_r, col, grid)
        reach_b0 = ~np.isnan(env_dla["b0"]).any(0)
        reach_g2 = ~np.isnan(env_dla["g2"]).any(0)
        ax.plot(grid[reach_b0], env_dla["b0"][:, reach_b0].mean(0), color="tab:blue", lw=2.0, ls="-",
                label="char-CNN -> DistilBERT (single threshold)")
        ax.fill_between(grid[reach_b0], env_dla["b0"][:, reach_b0].min(0), env_dla["b0"][:, reach_b0].max(0),
                        color="tab:blue", alpha=.13)
        ax.plot(grid[reach_g2], env_dla["g2"][:, reach_g2].mean(0), color="tab:blue", lw=1.7, ls="--",
                label="char-CNN -> DistilBERT (class-wise threshold)")

        env_e14 = GF.envelopes(paths_e14, t_front_14, t_r_14, col, grid_14)
        reach_14 = ~np.isnan(env_e14["b0"]).any(0)
        ax.plot(grid_14[reach_14], env_e14["b0"][:, reach_14].mean(0), color="tab:green", lw=1.8, ls="-",
                label="Early-exit -> DistilBERT (single threshold)")

        s_dla = dla["e10_3"][view_key]["pooled_speedup"]["b0"]
        f_dla = dla["e10_3"][view_key]["b0"]["macro_f1"]
        ax.scatter([s_dla], [f_dla], marker="D", s=105, color="tab:blue", edgecolors="k", linewidths=.9,
                   zorder=6, label="Operating point: char-CNN -> DistilBERT")

        row_14 = e14_views[view_key]
        s_e14 = min(93.0, dl_a.pooled_speedup(row_14["per_fold_speedup"]["b0"]))
        f_e14 = row_14["b0"]["macro_f1"]
        ax.scatter([s_e14], [f_e14], marker="D", s=95, color="tab:green", edgecolors="k", linewidths=.9,
                   zorder=6, label="Operating point: Early-exit -> DistilBERT")

        ax.axhline(f1_bert, color="tab:purple", lw=1.1, ls="-.", alpha=.8)
        ax.scatter([1.0], [f1_bert], marker="h", s=115, color="tab:purple", edgecolors="k", linewidths=.8,
                   zorder=6, label="DistilBERT alone (reference, 1.0x)")
        ax.axhline(f1_cnn, color="k", lw=1.0, ls=":", alpha=.6)
        ax.scatter([max_speed], [f1_cnn], marker="*", s=175, color="k", zorder=6,
                   label=f"char-CNN alone ({max_speed:.1f}x)")

        ax.set_xlim(0, 95)
        ax.set_ylim(0.980, 0.996)
        ax.set_xlabel("Speedup vs. DistilBERT alone (attack share 3.3%)")
        ax.set_title(f"({letter}) Same data & CPU (batch 1) | {view_title}", fontsize=10)
        ax.grid(alpha=.3)
    axes[1].set_ylabel("Test Macro-F1 (5-fold CV, shaded = fold min-max)")
    legends.append(axes[1].legend(fontsize=8.2, loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=2, frameon=False))
    for legend in legends:
        legend.set_in_layout(False)
    title = fig.suptitle(
        "Accuracy-latency trade-off of deep-learning configurations on SR-BH 2020 (5-fold CV, CPU batch 1)",
        fontsize=12,
    )
    fig.tight_layout()
    FIGURE_DL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DL, dpi=140, bbox_inches="tight", bbox_extra_artists=[*legends, title])
    plt.close(fig)
    return FIGURE_DL, ours_a


def dl_cascade_points(view):
    """(라벨, char-CNN 대비 배수 S, fold-test Macro-F1, 표식, 색). 파일이 없으면(E8 미실행) 빈 목록."""
    if not DL_CASCADE.exists():
        return []
    result = json.loads(DL_CASCADE.read_text(encoding="utf-8"))
    points = [("char-CNN early exit at k3 (ours)", result["dl_c"][view]["b0"]["speedup"],
               result["dl_c"][view]["b0"]["macro_f1"], "D", "darkgoldenrod")]
    if "k3_alone" in result:
        points.append(("k3-only char-CNN alone (ours, control)", result["k3_alone"]["speedup"],
                       result["k3_alone"]["macro_f1"], "s", "tab:gray"))
    if DL_A.exists():
        # E10-3 운영점. 배수는 같은 실행의 char-CNN 지연 기준(t2 / (t2 + Ē_π·t_R)) — 이 그림의 다른 점과 같은 기준이다.
        # 평균 비용(합산, §E11)을 쓴다: fold 산술평균은 넘김이 많은 fold 4 를 거의 무시해 점을 실제보다 빠르게 찍는다.
        row = json.loads(DL_A.read_text(encoding="utf-8"))["e10_3"][view]
        if "reason" not in row:
            points.append(("char-CNN -> DistilBERT cascade (ours)", row["vs_charcnn"]["latency_ratio_pooled"],
                           row["b0"]["macro_f1"], "D", "tab:blue"))
    return points


def distilbert_reimpl():
    """같은 fold 로 재구현한 WAMM DistilBERT 단독(E10-2 의 R): (char-CNN 대비 배수, Macro-F1). 없으면 None."""
    if not DL_A.exists():
        return None
    result = json.loads(DL_A.read_text(encoding="utf-8"))
    ms = result["latency_ms"]
    return ms["t2"] / ms["t_R"], result["distilbert_macro_f1"]


def stage1_styles(dl_only):
    """그림에 올릴 1차 목록. 딥러닝만 남기는 판은 선형 1차 둘을 빼고, 빈 자리에 E2 의 나머지 후보(최대값 풀링)를 넣는다."""
    if not dl_only:
        return STAGE1_STYLE
    return {**{name: style for name, style in STAGE1_STYLE.items() if name not in CLASSICAL_STAGE1},
            "cnn_gmp": ("tab:pink", "RGB CNN global max pool (ours)")}


def ours_reported_style(analysis, styles, view="val_mix"):
    """우리 결과를 (a) 패널의 형식(요청당 지연 ms, Macro-F1)으로 바꾼다. 운영점을 못 고른 구성은 뺀다."""
    ms2 = analysis["stage2_ms"]
    points = [("char-CNN alone (ours, stage 2)", ms2, analysis["stage2_solo_macro_f1"], "*", "k")]
    for name, row in analysis["stage1"].items():
        if name not in styles:
            continue
        color, label = styles[name]
        points.append((f"{label} alone", row["ms1"], row["solo_macro_f1"], "s", color))
        if "reason" not in row[view]:
            # 배수 S = t2 / (기대 지연) 이므로 기대 지연 = t2 / S.
            points.append((f"{label} -> char-CNN cascade", ms2 / row[view]["b0"]["speedup"],
                           row[view]["b0"]["macro_f1"], "D", color))
    if "rgb_cnn" in styles and "tfidflr" not in styles:     # 딥러닝 판(stage1_styles(dl_only=True))에서만
        points += [(label, ms2 / speedup, score, marker, color)
                   for label, speedup, score, marker, color in dl_cascade_points(view)]
        if distilbert_reimpl():
            speedup, score = distilbert_reimpl()
            points.append(("DistilBERT alone (WAMM'25 re-impl., same folds)", ms2 / speedup, score, "h", "tab:blue"))
    return points


def plot(analysis, loaded, styles, dl_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms2 = analysis["stage2_ms"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    for label, ms, score, marker, color, is_f1 in (REPORTED_DL if dl_only else REPORTED):
        ax.scatter([ms], [score], marker=marker, s=90, zorder=4, label=label,
                   facecolors=color if is_f1 else "none", edgecolors=color)
    for label, ms, score, marker, color in ours_reported_style(analysis, styles):
        ax.scatter([ms], [score], marker=marker, s=170 if marker == "*" else 80, color=color, zorder=5,
                   edgecolors="k", linewidths=.8, label=label + " (SR-BH, CPU)")
    for label, score, color in NO_LATENCY:
        ax.axhline(score, color=color, lw=1, ls="--", label=label)
    ax.set_xscale("log")
    ax.invert_xaxis()                       # 다른 패널처럼 오른쪽이 빠른 쪽이 되게 한다
    ax.set_ylim(0.74 if dl_only else 0.88, 1.001)   # WAMM 의 딥러닝 보고값(0.77·0.85)이 잘리지 않게
    ax.set_xlabel("Latency per request, ms (log scale; faster to the right)")
    ax.set_ylabel("F1 as reported (hollow marker = accuracy)")
    ax.set_title("(a) As reported: different data, classes and hardware (not directly comparable)", fontsize=10)
    ax.grid(alpha=.3, which="both")
    # 점이 아래쪽까지 퍼져 있어 범례를 축 안에 두면 점을 가린다 — 축 아래로 뺀다.
    legends = [ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, frameon=False)]

    grid = np.linspace(0.1, max(ms2 / analysis["stage1"][name]["ms1"] for name in loaded), 300)
    for ax, (col, (_, title)), letter in zip(axes[1:], enumerate(GF.NORMAL_VIEWS), "bc"):
        for name, L in loaded.items():
            color, label = styles[name]
            ms1 = analysis["stage1"][name]["ms1"]
            env = GF.envelopes(L["paths"], ms1, ms2, col, grid)
            for gate, style, gate_label in (("b0", "-", "standard deferral"), ("g2", "--", "class-wise tau")):
                reachable = ~np.isnan(env[gate]).any(0)
                ax.plot(grid[reachable], env[gate][:, reachable].mean(0), color=color, lw=2, ls=style,
                        label=f"stage 1 = {label}, {gate_label}")
            reachable = ~np.isnan(env["b0"]).any(0)
            ax.fill_between(grid[reachable], env["b0"][:, reachable].min(0), env["b0"][:, reachable].max(0),
                            color=color, alpha=.12)
            ax.scatter([ms2 / ms1], [L["solo"][0]], marker="s", s=60, color=color, zorder=5)
        ax.scatter([1.0], [analysis["stage2_solo_macro_f1"]], marker="*", s=180, c="k", zorder=6,
                   label="char-CNN alone")
        for label, speedup, score, marker, color in (dl_cascade_points(GF.NORMAL_VIEWS[col][0]) if dl_only else []):
            ax.scatter([speedup], [score], marker=marker, s=80, color=color, edgecolors="k", linewidths=.8,
                       zorder=6, label=label)
        if dl_only and distilbert_reimpl():
            # 배수가 0.01 대라 점으로 찍으면 가로축이 눌린다 — 정확도만 가로선으로 보인다.
            speedup, score = distilbert_reimpl()
            ax.axhline(score, color="tab:blue", lw=1, ls="-.",
                       label=f"DistilBERT alone (WAMM'25 re-impl., same folds, {speedup:.4f}x)")
        ax.axvline(1.0, color="k", lw=.8, ls=":")
        ax.set_ylim(0.88, 1.001)
        ax.set_xlabel("Speedup vs char-CNN alone (attack share 3.3%)")
        ax.set_title(f"({letter}) Same data, same CPU, batch 1 | {title}", fontsize=10)
        ax.grid(alpha=.3)
    axes[1].set_ylabel("Test Macro-F1 (5-fold CV, band = fold min-max)")
    legends.append(axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, frameon=False))
    for legend in legends:
        legend.set_in_layout(False)     # 축 밖 범례가 패널 간격을 벌리지 않게 한다(저장 때는 bbox 로 포함)
    title = fig.suptitle("Accuracy-speed comparison with the best reported similar studies "
                 + ("- deep-learning models only " if dl_only else "")
                 + "(squares = stage 1 alone, diamonds = cascade)", fontsize=12)
    fig.tight_layout()
    figure = FIGURE_DL if dl_only else FIGURE
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=130, bbox_inches="tight", bbox_extra_artists=[*legends, title])
    plt.close(fig)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dl-only", action="store_true", help="고전 ML 을 빼고 딥러닝 모델만 남긴 판을 그린다")
    args = parser.parse_args()
    try:
        strata = np.asarray(CV.load_pool_strata("srbh_4class", exclude_test=True))
        if args.dl_only:
            figure, rows = plot_dl_only(strata)
            for label, ms, score, _, _ in rows:
                print(f"{label}: {ms:.4f} ms, Macro-F1 {score:.4f}")
            print(f"그림: {figure.relative_to(PROJECT_ROOT).as_posix()}")
            return 0
        styles = stage1_styles(False)
        analysis = json.loads(LS.OUTPUT.read_text(encoding="utf-8"))
        paths = {"rgb_cnn": GF.STAGE1, **{name: LS.probs_path(name) for name in LS.CANDIDATES},
                 **{name: GF.RESULTS / f"cvprobs_srbh_4class_{name}_raw_rgb_sealed.npz" for name in LS.E2_LABELS}}
        loaded = {name: LS.load_stage1(paths[name], strata) for name in analysis["stage1"] if name in styles}
        figure = plot(analysis, loaded, styles, False)
        for label, ms, score, _, _ in ours_reported_style(analysis, styles):
            print(f"{label}: {ms:.4f} ms, Macro-F1 {score:.4f}")
        print(f"그림: {figure.relative_to(PROJECT_ROOT).as_posix()}")
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] 문헌 비교 그림 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
