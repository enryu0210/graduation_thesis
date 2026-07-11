"""
Phase 5 (RQ2) — 5모델 회피 취약성 통합 비교 그림 (헤드라인)

무엇을 하나:
    run_evasion.py 가 모델별로 저장한 evasion_{track}_{model}_stacked.json 5개를 모아
    "예산 k vs 회피 지표" 곡선을 **한 그림에 겹쳐** 그린다. RQ2 의 핵심 질문
    ("표현방식(이미지/시퀀스/TF-IDF)별 상대 취약성")을 한 장으로 답한다.

두 패널(docs/05 §2.2 의 두 지표를 분리해 보여준다):
    (좌) 주 지표 benign-evasion(공격→Normal) — WAF 우회. 5모델 모두 ~0 인지 한눈에.
    (우) 보조 지표 any-misclass(예측≠진짜) — 어떤 표현이 표면 변형에 더 '흔들리나'.

왜 필요한가:
    모델별 그림을 따로 보면 "누가 더 취약한가"를 비교하기 어렵다. 같은 축에 겹쳐야
    RQ1 clean 순위가 회피에서 뒤집히는지(가설 H1)를 육안으로 판정할 수 있다.

산출물:
    docs/figures/attacks/evasion_compare_{track}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "attacks"

# 표시 순서와 색 — 표현방식 묶음이 보이게: 이미지(빨강 단독) / 시퀀스(파랑계열) / TF-IDF(회색계열).
MODEL_STYLE = [
    ("cnn",          "proposed CNN (image)",   "#c0392b", "o-"),
    ("charcnn",      "char-CNN (sequence)",    "#2980b9", "s-"),
    ("bilstm",       "BiLSTM (sequence)",      "#8e44ad", "^-"),
    ("tfidf_logreg", "TF-IDF + LogReg",        "#7f8c8d", "D--"),
    ("tfidf_rf",     "TF-IDF + RandomForest",  "#2c3e50", "v--"),
]


def load_stacked(track: str, model: str):
    """모델별 stacked 결과를 (k리스트, benign-evasion, any-misclass) 로 로드한다."""
    path = RESULTS_DIR / f"evasion_{track}_{model}_stacked.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    ks = [r["budget_k"] for r in rows]
    be = [r["asr_mutated"] for r in rows]
    am = [r["anymis_mutated"] for r in rows]
    return ks, be, am


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description="RQ2 5모델 회피 취약성 통합 비교 그림")
    p.add_argument("--track", default="payload_4class_csicnorm")
    args = p.parse_args()

    fig, (ax_be, ax_am) = plt.subplots(1, 2, figsize=(13, 5))
    found = []
    for model, label, color, style in MODEL_STYLE:
        loaded = load_stacked(args.track, model)
        if loaded is None:
            print(f"  [건너뜀] 결과 없음: {model}")
            continue
        ks, be, am = loaded
        ax_be.plot(ks, be, style, color=color, lw=2, ms=6, label=label)
        ax_am.plot(ks, am, style, color=color, lw=2, ms=6, label=label)
        found.append((label, be[-1], am[-1]))

    # (좌) 주 지표 — 전부 ~0 임을 보이되, 0 근방 미세차를 보이도록 y 상한을 작게.
    ax_be.set_title("Primary: benign-evasion (attack -> Normal = WAF bypass)", fontsize=11)
    ax_be.set_xlabel("stacked mutation budget k")
    ax_be.set_ylabel("benign-evasion rate")
    ax_be.set_ylim(-0.002, 0.05)
    ax_be.axhline(0, color="gray", lw=0.8, ls=":")
    ax_be.legend(loc="upper left", fontsize=9)

    # (우) 보조 지표 — 표현방식별 '동요' 차이가 드러나는 패널.
    ax_am.set_title("Secondary: any-misclass (pred != true class)", fontsize=11)
    ax_am.set_xlabel("stacked mutation budget k")
    ax_am.set_ylabel("any-misclass rate")
    ax_am.set_ylim(-0.02, 1.02)
    ax_am.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"RQ2 evasion vulnerability across 5 models ({args.track})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f"evasion_compare_{args.track}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"[저장] 통합 비교 그림 → {out}")
    print("  모델별 (k=5) benign-evasion / any-misclass:")
    for label, be5, am5 in found:
        print(f"    {label:<26} BE={be5:.4f}  AM={am5:.4f}")


if __name__ == "__main__":
    main()
