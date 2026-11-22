"""E17 — SR-BH 경로 명령 삽입 라벨 특이점: 원 라벨 오답 분해(--audit)와 보정 라벨 판정(--analyze).

배경(docs/14 §8.45): SR-BH 는 같은 셸 명령 삽입도 URL 경로에 넣으면 SQLi, 쿼리·바디에 넣으면 CmdI 로 라벨링했다.
그래서 "SQLi→CmdI 오탐"의 상당수는 모델이 의미대로 맞힌 요청이다.
  --audit   : (학습 없음) 원 라벨 cvprobs 의 SQLi→CmdI 오답을 유형별로 나누고, 보정 행 목록을 docs/audit 에 남긴다.
  --analyze : 보정 라벨(_rlp) CV 의 char-CNN·DistilBERT·캐스케이드(B0)를 비교해 E17-2·E17-3 을 판정한다.
규칙·판정 기준은 docs/paper_revision/03_prereg_v2.md §E17 에 학습 전에 고정했다.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "src" / "models"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from src.data.srbh_track import CMD_PATTERN, SQL_PATTERN, _decode, path_cmdi_mask  # noqa: E402
from src.eval import cross_validate  # noqa: E402
from src.eval import dl_a  # noqa: E402
from src.eval import dl_cascade as DC  # noqa: E402
from src.eval import gate_confirm as GC  # noqa: E402
from src.eval.dl_a_errors import cascade_predictions, pooled_test  # noqa: E402

RESULTS = PROJECT_ROOT / "experiments" / "results"
AUDIT_CSV = PROJECT_ROOT / "docs" / "audit" / "srbh_path_cmdi_relabel.csv"
SPLITS = ("train", "val", "test")
CMDI, SQLI = "CommandInjection", "SQLInjection"


def load_frames(splits) -> pd.DataFrame:
    from data_text import csv_path

    return pd.concat([pd.read_csv(csv_path("srbh_4class", s), encoding="utf-8", dtype=str, keep_default_na=False)
                      .assign(split=s) for s in splits], ignore_index=True)


def error_kind(text: str, other_fields: str) -> str:
    """오답 한 행을 '왜 틀렸나' 유형으로 — 경로 주입 특이점 / 입력 밖(UA·쿠키) / 입력에 흔적 없음 등."""
    if path_cmdi_mask([text], [SQLI])[0]:
        return "path_cmdi"
    decoded = _decode(text)
    if SQL_PATTERN.search(decoded):
        return "sql_in_input"
    if CMD_PATTERN.search(decoded):
        return "cmd_elsewhere"
    hidden = _decode(other_fields)
    if SQL_PATTERN.search(hidden) or CMD_PATTERN.search(hidden):
        return "attack_outside_f2"
    return "no_pattern"


def audit() -> dict:
    # 1) 보정 행 목록(전 split) — 추적되는 감사 파일로 남겨 다른 기기·심사에서 같은 행을 재확인할 수 있게 한다.
    full = load_frames(SPLITS)
    mask = np.array(path_cmdi_mask(full.text_raw.tolist(), full.label.tolist()))
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.loc[mask, ["row_id", "split"]].to_csv(AUDIT_CSV, index=False, encoding="utf-8")
    path_cmd = [bool(CMD_PATTERN.search(_decode(t.split("\n", 1)[0].partition("?")[0]))) for t in full.text_raw]
    by_label = {lab: {"rows": int((full.label == lab).sum()),
                      "path_cmd": int(np.sum((full.label == lab).to_numpy() & np.array(path_cmd))),
                      "relabeled": int(np.sum((full.label == lab).to_numpy() & mask))}
                for lab in sorted(full.label.unique())}

    # 2) 원 라벨 CV(train+val 풀) 오답 분해. 풀 순서는 load_text_pool 과 같아야 한다 — 어긋나면 엉뚱한 행을 센다.
    pool = load_frames(SPLITS[:2])
    texts, y, classes = cross_validate.load_text_pool("srbh_4class", "raw", exclude_test=True)
    if pool.text_raw.tolist() != texts:
        raise ValueError("CSV 프레임과 CV 풀의 행 순서가 다릅니다.")
    others = (pool.request_cookie + "\n" + pool.request_user_agent).tolist()
    c, s = classes.index(CMDI), classes.index(SQLI)
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    errors = {}
    for seed in (42, 43):
        solo = DC.load_cvprobs("charcnn", seed=seed)
        bert = dl_a.load_npz(dl_a.distilbert_cvprobs(seed))
        DC.check_aligned(solo, bert)
        idx, yt, p1 = pooled_test(solo)
        cascade, _ = cascade_predictions(solo, bert, strata)
        preds = {"charcnn": p1.argmax(1), "distilbert": pooled_test(bert)[2].argmax(1), "cascade_b0": cascade}
        for name, pred in preds.items():
            rows = idx[(yt == s) & (pred == c)]
            back = idx[(yt == c) & (pred == s)]
            errors[f"s{seed}_{name}"] = {
                "sqli_as_cmdi": int(len(rows)),
                "sqli_as_cmdi_kinds": dict(collections.Counter(error_kind(texts[i], others[i]) for i in rows)),
                "cmdi_as_sqli": int(len(back)),
                "cmdi_as_sqli_kinds": dict(collections.Counter(error_kind(texts[i], others[i]) for i in back))}
    pool_mask = np.array(path_cmdi_mask(texts, [classes[i] for i in y]))
    return {"by_label_all_splits": by_label, "relabeled_all_splits": int(mask.sum()),
            "relabeled_cv_pool": int(pool_mask.sum()), "cv_pool_n": int(len(y)), "errors_original_labels": errors}


# --------------------------------------------------------------------------
# --analyze : 보정 라벨 판정
# --------------------------------------------------------------------------
def relabeled_paths(seed: int) -> tuple[Path, Path]:
    suffix = DC.cvprobs_suffix(seed)
    return (RESULTS / f"cvprobs_srbh_4class_charcnn_raw_rlp_sealed{suffix}.npz",
            RESULTS / f"cvprobs_srbh_4class_distilbert_raw_lr2e-05_rlp_sealed{suffix}.npz")


def fold_metrics(y, pred, c, s, n_classes) -> dict:
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (y, pred), 1)
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return {"macro_f1": float(f1.mean()), "cmdi_f1": float(f1[c]),
            "cmdi_precision": float(cm[c, c] / max(cm[:, c].sum(), 1)), "cmdi_recall": float(cm[c, c] / max(cm[c].sum(), 1)),
            "sqli_as_cmdi": int(cm[s, c]), "cmdi_as_sqli": int(cm[c, s])}


def analyze(seed: int) -> dict:
    solo_path, bert_path = relabeled_paths(seed)
    solo, bert = dl_a.load_npz(solo_path), dl_a.load_npz(bert_path)
    DC.check_aligned(solo, bert)
    classes = solo["classes"].tolist()
    c, s = classes.index(CMDI), classes.index(SQLI)
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    idx, y, p1 = pooled_test(solo)
    cascade, routed = cascade_predictions(solo, bert, strata)
    preds = {"charcnn": p1.argmax(1), "distilbert": pooled_test(bert)[2].argmax(1), "cascade_b0": cascade}
    fold_of = np.empty(len(solo["y"]), int)
    for k in range(1, 6):
        fold_of[solo[f"test_idx_{k}"]] = k
    folds = fold_of[idx]
    per_fold = {name: [fold_metrics(y[folds == k], pred[folds == k], c, s, len(classes)) for k in range(1, 6)]
                for name, pred in preds.items()}
    pooled = {name: fold_metrics(y, pred, c, s, len(classes)) for name, pred in preds.items()}

    def contrast(model: str, key: str) -> dict:
        d = np.array([a[key] - b[key] for a, b in zip(per_fold[model], per_fold["charcnn"])])
        return {"mean_pp": float(d.mean() * 100), "per_fold_pp": (d * 100).round(4).tolist(), **GC.corrected_t(d)}

    return {"seed": seed, "n": int(len(y)), "classes": classes, "n_cmdi": int(np.sum(y == c)),
            "routed_rate_test": float(routed.mean()), "pooled": pooled, "per_fold": per_fold,
            # E17-2·E17-3 판정 대상 = CmdI F1. Macro-F1 은 병기.
            "e17_2_distilbert_vs_charcnn": {"cmdi_f1": contrast("distilbert", "cmdi_f1"),
                                            "macro_f1": contrast("distilbert", "macro_f1")},
            "e17_3_cascade_vs_charcnn": {"cmdi_f1": contrast("cascade_b0", "cmdi_f1"),
                                         "macro_f1": contrast("cascade_b0", "macro_f1")}}


def verdict(rows: dict) -> dict:
    """§E17: 두 seed 모두 평균 차 > 0 이고 seed 42 보정 t 양측 p < 0.05 → '보정 라벨에서도 CmdI 이득 유지'."""
    out = {}
    for key in ("e17_2_distilbert_vs_charcnn", "e17_3_cascade_vs_charcnn"):
        stats = {seed: row[key]["cmdi_f1"] for seed, row in rows.items()}
        positive = all(v["mean_pp"] > 0 for v in stats.values())
        base = stats.get("s42", {}).get("p")
        out[key] = {"all_seeds_positive": positive, "s42_p": base,
                    "holds": bool(positive and base is not None and base < 0.05 and len(stats) == 2)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    try:
        if args.audit:
            result, output = audit(), RESULTS / "relabel_audit_srbh_4class.json"
            print(f"보정 행: 전 split {result['relabeled_all_splits']:,} · CV 풀 {result['relabeled_cv_pool']:,}"
                  f" / {result['cv_pool_n']:,} → {AUDIT_CSV.relative_to(PROJECT_ROOT).as_posix()}")
            for lab, row in result["by_label_all_splits"].items():
                print(f"  {lab}: {row['rows']:,}행 · 경로 셸 명령 {row['path_cmd']:,} · 보정 {row['relabeled']:,}")
            for name, row in result["errors_original_labels"].items():
                print(f"  [{name}] SQLi→CmdI {row['sqli_as_cmdi']} {row['sqli_as_cmdi_kinds']}"
                      f" | CmdI→SQLi {row['cmdi_as_sqli']} {row['cmdi_as_sqli_kinds']}")
        else:
            rows = {f"s{seed}": analyze(seed) for seed in args.seeds}
            result = {**rows, "verdict": verdict(rows)}
            output = RESULTS / "relabel_eval_srbh_4class.json"
            for key, row in rows.items():
                print(f"[{key}] n={row['n']:,} CmdI {row['n_cmdi']:,} · 넘김(test) {row['routed_rate_test'] * 100:.2f}%")
                for name, m in row["pooled"].items():
                    print(f"   {name}: Macro-F1 {m['macro_f1']:.4f} · CmdI F1 {m['cmdi_f1']:.4f}"
                          f" (P {m['cmdi_precision']:.4f} R {m['cmdi_recall']:.4f})"
                          f" · SQLi→CmdI {m['sqli_as_cmdi']} · CmdI→SQLi {m['cmdi_as_sqli']}")
                for key2 in ("e17_2_distilbert_vs_charcnn", "e17_3_cascade_vs_charcnn"):
                    for metric, v in row[key2].items():
                        p = "-" if v["p"] is None else f"{v['p']:.3f}"
                        print(f"   {key2} {metric}: {v['mean_pp']:+.3f}%p (p {p})")
            print(f"판정: {result['verdict']}")
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] E17 실패: {error}", file=sys.stderr)
        return 1
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {output.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
