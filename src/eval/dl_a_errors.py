"""E11 사후 분석(병기) — char-CNN 단독과 DistilBERT 단독은 어떤 오답이 다른가.

판정이 아니다. E11(§8.35)이 "DistilBERT 가 seed 43 에서 강해져 캐스케이드가 따라가지 못했다"로 끝났으므로,
DistilBERT 가 더 잘 잡는 것이 '공격 여부'(차단)인지 '공격 종류'인지를 기존 cvprobs 만으로 가른다(학습 없음).
5 fold 의 test 예측을 풀 인덱스로 합쳐(각 행은 정확히 한 fold 의 test) 혼동 행렬·층별 오탐·오답 교차를 낸다.
결과는 docs/14 §8.36. 결과를 보고 고른 분석이라 E11 판정을 뒤집는 데 쓰지 않는다.
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
from src.eval import gate_frontier as GF  # noqa: E402

OUTPUT = GF.RESULTS / "dl_a_errors_srbh_4class.json"
# char-CNN 이 '확신했는데 틀린' 오답은 넘김으로 고칠 수 없다 — 이 문턱 이상을 그런 오답으로 센다.
CONFIDENT = 0.99


def pooled_test(probs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fold-test 확률을 풀 인덱스 순서로 합친다 → (풀 인덱스, 정답, 확률)."""
    idx = np.concatenate([probs[f"test_idx_{k}"] for k in range(1, 6)])
    p = np.concatenate([probs[f"test_probs_{k}"] for k in range(1, 6)])
    order = np.argsort(idx)
    return idx[order], probs["y"][idx[order]], p[order]


def compare(y, prob_a, prob_b, strata, classes) -> dict:
    """a = char-CNN(1차), b = DistilBERT(2차). strata 는 각 행의 Normal 출처 층(공격 행은 무시)."""
    pred_a, pred_b = prob_a.argmax(1), prob_b.argmax(1)
    normal = classes.index("Normal")
    attack = y != normal

    def confusion(pred):
        cm = np.zeros((len(classes), len(classes)), int)
        np.add.at(cm, (y, pred), 1)
        return cm.tolist()

    def summary(pred):
        return {"missed_attack": int(np.sum(attack & (pred == normal))),          # 차단 실패
                "false_alarm": int(np.sum(~attack & (pred != normal))),           # 정상 오탐
                "attack_type_confusion": int(np.sum(attack & (pred != normal) & (pred != y))),  # 차단은 됨
                "confusion": confusion(pred)}

    wrong_a, wrong_b = pred_a != y, pred_b != y
    fixed = wrong_a & ~wrong_b
    conf_a = prob_a.max(1)
    strata_fpr = {}
    for name in np.unique(strata[~attack]):
        m = ~attack & (strata == name)
        strata_fpr[str(name)] = {"n": int(m.sum()), "a": float(np.mean(pred_a[m] != normal)),
                                 "b": float(np.mean(pred_b[m] != normal))}
    return {"classes": classes, "charcnn": summary(pred_a), "distilbert": summary(pred_b),
            "normal_fpr_by_stratum": strata_fpr,
            "both_wrong": int(np.sum(wrong_a & wrong_b)), "only_charcnn_wrong": int(fixed.sum()),
            "only_distilbert_wrong": int(np.sum(~wrong_a & wrong_b)),
            # DistilBERT 만 맞힌 요청을 char-CNN 이 얼마나 확신했나 — 낮으면 넘김으로 되찾을 여지가 있다.
            "fixed_charcnn_conf_median": float(np.median(conf_a[fixed])) if fixed.any() else None,
            "fixed_charcnn_conf_ge_confident": float(np.mean(conf_a[fixed] >= CONFIDENT)) if fixed.any() else None,
            "all_charcnn_conf_ge_confident": float(np.mean(conf_a >= CONFIDENT))}


def analyze(seed: int) -> dict:
    solo = DC.load_cvprobs("charcnn", seed=seed)
    bert = dl_a.load_npz(GF.RESULTS / dl_a.DISTILBERT_CVPROBS.format(DC.cvprobs_suffix(seed)))
    DC.check_aligned(solo, bert)
    idx, y, prob_a = pooled_test(solo)
    _, _, prob_b = pooled_test(bert)
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))[idx]
    return {"seed": seed, "n": int(len(y)), **compare(y, prob_a, prob_b, strata, solo["classes"].tolist())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    try:
        result = {f"s{seed}": analyze(seed) for seed in args.seeds}
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] DL-A 오답 분석 실패: {error}", file=sys.stderr)
        return 1
    for key, row in result.items():
        a, b = row["charcnn"], row["distilbert"]
        print(f"[{key}] char-CNN → DistilBERT  차단 실패 {a['missed_attack']} → {b['missed_attack']}"
              f" · 정상 오탐 {a['false_alarm']} → {b['false_alarm']}"
              f" · 공격 종류 혼동 {a['attack_type_confusion']} → {b['attack_type_confusion']}")
        print(f"   오답 교차: 둘 다 {row['both_wrong']} · char-CNN만 {row['only_charcnn_wrong']}"
              f" · DistilBERT만 {row['only_distilbert_wrong']} | DistilBERT만 맞힌 요청의 char-CNN 확신도"
              f" 중앙값 {row['fixed_charcnn_conf_median']:.4f} · ≥{CONFIDENT} 비율 {row['fixed_charcnn_conf_ge_confident']:.3f}")
        for name, s in row["normal_fpr_by_stratum"].items():
            print(f"   Normal[{name}] n={s['n']} FPR {s['a'] * 100:.2f}% → {s['b'] * 100:.2f}%")
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {OUTPUT.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
