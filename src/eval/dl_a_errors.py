"""E11 사후 분석(병기) — char-CNN 단독과 DistilBERT 단독은 어떤 오답이 다른가. E12-2 의 기술 분석도 여기서 낸다.

판정이 아니다. E11(§8.35)이 "DistilBERT 가 seed 43 에서 강해져 캐스케이드가 따라가지 못했다"로 끝났으므로,
DistilBERT 가 더 잘 잡는 것이 '공격 여부'(차단)인지 '공격 종류'인지를 기존 cvprobs 만으로 가른다(학습 없음).
5 fold 의 test 예측을 풀 인덱스로 합쳐(각 행은 정확히 한 fold 의 test) 혼동 행렬·층별 오탐·오답 교차를 낸다.
결과는 docs/14 §8.36. 결과를 보고 고른 분석이라 E11 판정을 뒤집는 데 쓰지 않는다.

E12-2(`--max-tokens 512 128`, 사전 문서 §E12): 128 토큰판의 클래스별 F1·혼동 · **캐스케이드(B0 운영점) 예측의 혼동**
(DistilBERT 단독의 SQLi→CmdI 감소를 캐스케이드가 얼마나 물려받는가) · 128 에서 잘린 행과 안 잘린 행의 정답률.
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
from src.models import gate_signals as gates  # noqa: E402

OUTPUT = GF.RESULTS / "dl_a_errors_srbh_4class.json"
# char-CNN 이 '확신했는데 틀린' 오답은 넘김으로 고칠 수 없다 — 이 문턱 이상을 그런 오답으로 센다.
CONFIDENT = 0.99


def pooled_test(probs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fold-test 확률을 풀 인덱스 순서로 합친다 → (풀 인덱스, 정답, 확률)."""
    idx = np.concatenate([probs[f"test_idx_{k}"] for k in range(1, 6)])
    p = np.concatenate([probs[f"test_probs_{k}"] for k in range(1, 6)])
    order = np.argsort(idx)
    return idx[order], probs["y"][idx[order]], p[order]


def summary(y, pred, classes) -> dict:
    normal, attack = classes.index("Normal"), y != classes.index("Normal")
    cm = np.zeros((len(classes), len(classes)), int)
    np.add.at(cm, (y, pred), 1)
    # 클래스별 F1 = 2TP / (행 합 + 열 합).
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return {"missed_attack": int(np.sum(attack & (pred == normal))),          # 차단 실패
            "false_alarm": int(np.sum(~attack & (pred != normal))),           # 정상 오탐
            "attack_type_confusion": int(np.sum(attack & (pred != normal) & (pred != y))),  # 차단은 됨
            "sqli_as_cmdi": int(cm[classes.index("SQLInjection"), classes.index("CommandInjection")]),
            "class_f1": dict(zip(classes, f1.round(6).tolist())),
            "confusion": cm.tolist()}


def compare(y, prob_a, prob_b, strata, classes) -> dict:
    """a = char-CNN(1차), b = DistilBERT(2차). strata 는 각 행의 Normal 출처 층(공격 행은 무시)."""
    pred_a, pred_b = prob_a.argmax(1), prob_b.argmax(1)
    normal = classes.index("Normal")
    attack = y != normal
    wrong_a, wrong_b = pred_a != y, pred_b != y
    fixed = wrong_a & ~wrong_b
    conf_a = prob_a.max(1)
    strata_fpr = {}
    for name in np.unique(strata[~attack]):
        m = ~attack & (strata == name)
        strata_fpr[str(name)] = {"n": int(m.sum()), "a": float(np.mean(pred_a[m] != normal)),
                                 "b": float(np.mean(pred_b[m] != normal))}
    return {"classes": classes, "charcnn": summary(y, pred_a, classes), "distilbert": summary(y, pred_b, classes),
            "normal_fpr_by_stratum": strata_fpr,
            "both_wrong": int(np.sum(wrong_a & wrong_b)), "only_charcnn_wrong": int(fixed.sum()),
            "only_distilbert_wrong": int(np.sum(~wrong_a & wrong_b)),
            # DistilBERT 만 맞힌 요청을 char-CNN 이 얼마나 확신했나 — 낮으면 넘김으로 되찾을 여지가 있다.
            "fixed_charcnn_conf_median": float(np.median(conf_a[fixed])) if fixed.any() else None,
            "fixed_charcnn_conf_ge_confident": float(np.mean(conf_a[fixed] >= CONFIDENT)) if fixed.any() else None,
            "all_charcnn_conf_ge_confident": float(np.mean(conf_a >= CONFIDENT))}


def cascade_predictions(solo: dict, bert: dict, strata) -> tuple[np.ndarray, np.ndarray]:
    """dl_a 판정과 같은 B0 운영점(fold 마다 val 에서 고른 점)의 캐스케이드 예측 → (풀 순서 예측, 넘김 여부).

    운영점은 val 열만 보고 고르므로 view(전체·실사용자)와 지연 값에 무관하다 → 지연 자리에 1 을 넣는다.
    """
    paths, _ = GF.fold_paths(solo, bert, strata)
    row = GC.secondary_rule(paths, dl_a.fold_val_f1(bert), GC.R_REAL, 1.0, 1.0)
    if "reason" in row:
        raise ValueError(row["reason"])
    idx, preds, routed = [], [], []
    for k, point in zip(range(1, 6), row["per_fold_index"]["b0"]):
        p1, p2 = solo[f"test_probs_{k}"], bert[f"test_probs_{k}"]
        mask = p1.max(1) < GF.b0_taus(solo[f"val_probs_{k}"].max(1))[point]
        pred = np.where(mask, p2.argmax(1), p1.argmax(1))
        # 판정 파일과 같은 운영점인지 F1 로 맞춰 본다 — 어긋나면 혼동 행렬이 다른 캐스케이드의 것이 된다.
        y_test = solo["y"][solo[f"test_idx_{k}"]]
        if not np.isclose(gates._macro_f1_fast(y_test, pred, DC.N_CLASSES), row["per_fold_macro_f1"]["b0"][k - 1]):
            raise ValueError(f"fold {k}: 다시 만든 캐스케이드 예측이 운영점 F1 과 다르다")
        idx.append(solo[f"test_idx_{k}"])
        preds.append(pred)
        routed.append(mask)
    order = np.argsort(np.concatenate(idx))
    return np.concatenate(preds)[order], np.concatenate(routed)[order]


def token_lengths(pool_index: np.ndarray) -> np.ndarray:
    """풀 행의 DistilBERT 토큰 수(특수 토큰 포함, 자르기 없음) — 학습의 truncation(max_length=N)과 같은 셈."""
    from transformers import AutoTokenizer

    from src.models import text_models

    texts, _, _ = cross_validate.load_text_pool("srbh_4class", "raw", exclude_test=True)
    tok = AutoTokenizer.from_pretrained(text_models.DISTILBERT_NAME)
    ids = tok([texts[i] for i in pool_index], truncation=False)["input_ids"]
    return np.array([len(row) for row in ids])


def analyze(seed: int, tokens: tuple[int, ...]) -> dict:
    solo = DC.load_cvprobs("charcnn", seed=seed)
    berts = {n: dl_a.load_npz(dl_a.distilbert_cvprobs(seed, n)) for n in tokens}
    DC.check_aligned(solo, *berts.values())
    idx, y, prob_a = pooled_test(solo)
    classes = solo["classes"].tolist()
    pool_strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    probs = {n: pooled_test(b)[2] for n, b in berts.items()}
    # 기존(E11) 출력 형식: char-CNN vs DistilBERT 512.
    result = {"seed": seed, "n": int(len(y)),
              **compare(y, prob_a, probs[dl_a.FULL_TOKENS], pool_strata[idx], classes)}

    # E12-2: 토큰 길이별 단독 · 캐스케이드(B0) 예측. 넘김률은 test 전체 기준(배포 π 재가중 아님).
    result["by_tokens"] = {}
    for n, bert in berts.items():
        cascade, routed = cascade_predictions(solo, bert, pool_strata)
        result["by_tokens"][str(n)] = {"distilbert": summary(y, probs[n].argmax(1), classes),
                                       "cascade_b0": {**summary(y, cascade, classes),
                                                      "routed_rate": float(routed.mean())}}
    short = [n for n in tokens if n != dl_a.FULL_TOKENS]
    if short:
        lengths = token_lengths(idx)
        preds = {"charcnn": prob_a.argmax(1), **{f"distilbert_{n}": probs[n].argmax(1) for n in tokens}}
        for n in short:
            cut = lengths > n
            rows = {}
            for name, pred in preds.items():
                rows[name] = {"truncated_acc": float(np.mean(pred[cut] == y[cut])),
                              "kept_acc": float(np.mean(pred[~cut] == y[~cut]))}
            result["by_tokens"][str(n)]["truncated_rows"] = {
                "n_truncated": int(cut.sum()), "n_kept": int((~cut).sum()),
                "truncated_by_class": {c: int(np.sum(cut & (y == i))) for i, c in enumerate(classes)},
                "accuracy": rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[dl_a.FULL_TOKENS],
                        help="E12-2 는 512 128 (512 는 E11 비교의 기준이라 반드시 포함)")
    args = parser.parse_args()
    if dl_a.FULL_TOKENS not in args.max_tokens:
        parser.error(f"--max-tokens 에 {dl_a.FULL_TOKENS} 이 있어야 합니다(char-CNN 대비 비교의 기준).")
    tokens = tuple(dict.fromkeys(args.max_tokens))
    try:
        result = {f"s{seed}": analyze(seed, tokens) for seed in args.seeds}
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
        print(f"   char-CNN 단독: SQLi→CmdI {a['sqli_as_cmdi']} · CmdI F1 {a['class_f1']['CommandInjection']:.4f}")
        for n, block in row["by_tokens"].items():
            for name in ("distilbert", "cascade_b0"):
                s = block[name]
                extra = f" · 넘김(test 전체) {s['routed_rate'] * 100:.1f}%" if "routed_rate" in s else ""
                print(f"   [{n} 토큰] {name}: SQLi→CmdI {s['sqli_as_cmdi']} · 종류 혼동 {s['attack_type_confusion']}"
                      f" · 차단 실패 {s['missed_attack']} · CmdI F1 {s['class_f1']['CommandInjection']:.4f}{extra}")
            if "truncated_rows" in block:
                t = block["truncated_rows"]
                print(f"   [{n} 토큰] 잘린 행 {t['n_truncated']:,} ({t['truncated_by_class']})")
                for name, acc in t["accuracy"].items():
                    print(f"      {name}: 잘린 행 정답률 {acc['truncated_acc']:.4f} · 안 잘린 행 {acc['kept_acc']:.4f}")
    suffix = "".join(f"_tok{n}" for n in tokens if n != dl_a.FULL_TOKENS)
    output = OUTPUT.with_name(f"{OUTPUT.stem}{suffix}.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {output.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
