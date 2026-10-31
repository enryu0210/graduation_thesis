"""E15 — DL-A(char-CNN → DistilBERT)의 비용 기반(slowdown) 공격면. 판정 규칙은 03_prereg_v2.md §E15.

G5(gate_g5_mutation)와 같은 변형을 fold-test 공격 행에 걸고, `_ckpt` 실행이 저장한 fold 별 char-CNN 으로
변형 입력의 1차 확률을 다시 낸다. 운영점(B0·G2 τ)은 같은 실행의 val 확률 + DistilBERT seed 42 val 확률로
고른 값을 그대로 쓴다 — 변형 데이터로 재튜닝하지 않는다.

    python src/eval/dl_a_slowdown.py            # 전체(저장)
    python src/eval/dl_a_slowdown.py --limit 50 # fold 마다 공격 50행만, 저장 안 함(스모크)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cross_validate as CV  # noqa: E402  (src/models·src/eval 를 sys.path 에 올린다)
from src.eval import dl_a  # noqa: E402
from src.eval import dl_cascade as DC  # noqa: E402
from src.eval import gate_frontier as GF  # noqa: E402
from src.eval.dl_a_followup import e12_latency, operating  # noqa: E402
from src.eval.gate_g5_mutation import GATES, INFLATION_RATIO_LIMIT, LABEL_MUTATIONS, mutate_rows  # noqa: E402
from src.eval.gate_g6_cv import gate_mask  # noqa: E402
from src.models import gate_signals as gates  # noqa: E402

SEED = 42
CKPT_TAG = "srbh_4class_charcnn_raw_sealed_ckpt"
MIN_ROWS = 100       # §E15: 바뀐 행이 100행 이상인 변형만 판정 대상
MAX_LEN = 48 * 48    # cross_validate --max-len 기본값(char-CNN 학습 입력 길이)


def load_fold_net(k: int, test_idx: np.ndarray, device):
    import train as T
    saved = torch.load(PROJECT_ROOT / "experiments/checkpoints/cv" / CKPT_TAG / f"fold{k}.pt",
                       map_location=device, weights_only=False)
    if not np.array_equal(saved["test_idx"], test_idx):
        raise ValueError(f"fold{k}.pt 의 test 인덱스가 cvprobs 와 다릅니다(다른 실행의 가중치).")
    net = T.build_model(saved["model"], DC.N_CLASSES).to(device)
    net.load_state_dict(saved["state_dict"])
    return net.eval()


def predict(net, texts: list[str], device) -> np.ndarray:
    x = torch.from_numpy(CV.encode_byte_matrix_int16(texts, MAX_LEN).astype(np.int64))
    with torch.no_grad():
        return np.concatenate([torch.softmax(net(xb.to(device)), 1).cpu().numpy() for xb in x.split(512)])


def fold_points(first: dict, second: dict, k: int, index: dict) -> dict:
    """secondary_rule 이 고른 경로 번호 → 실제 운영점(B0 τ 하나 · G2 클래스별 τ)."""
    y_val, p1v = first["y"][first[f"val_idx_{k}"]], first[f"val_probs_{k}"]
    b0 = {"tau": GF.b0_taus(p1v.max(1))[index["b0"][k - 1]]}
    g2 = gates.classwise_tau_path(p1v, second[f"val_probs_{k}"], y_val, DC.N_CLASSES, GF.STEP)[index["g2"][k - 1]]
    return {"b0": b0, "g2": {"taus": g2["taus"]}}


def analyze(limit: int | None) -> dict:
    ms = e12_latency()
    t2, t_r = ms["t2"], ms["t_R"]
    with np.load(GF.RESULTS / f"cvprobs_{CKPT_TAG}.npz", allow_pickle=False) as data:
        first = dict(data)   # 기존 cvprobs 가 아니라 가중치를 저장한 그 실행의 확률(§E15)
    second = dl_a.load_npz(dl_a.distilbert_cvprobs(SEED))
    DC.check_aligned(first, second)
    row = operating(first, second, t2, t_r)
    texts, y, classes = CV.load_text_pool("srbh_4class", "raw", exclude_test=True)
    if not np.array_equal(y, first["y"]):
        raise ValueError("풀 라벨이 cvprobs 와 다릅니다(행 순서 불일치).")
    normal = classes.index("Normal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = list(dict.fromkeys(m for ms_ in LABEL_MUTATIONS.values() for m in ms_))
    # 변형별 5 fold 합산 카운트: 바뀐 행 수, 게이트별 넘김 수(원본·변형), 1차 공격 판정 수(원본·변형).
    tally = {n: {"n": 0, "clean": dict.fromkeys(GATES, 0), "mut": dict.fromkeys(GATES, 0),
                 "attack_clean": 0, "attack_mut": 0} for n in names}
    points, max_drift = [], 0.0
    for k in range(1, 6):
        test_idx = first[f"test_idx_{k}"]
        net = load_fold_net(k, test_idx, device)
        point = fold_points(first, second, k, row["per_fold_index"])
        points.append(point)
        attack = np.flatnonzero(first["y"][test_idx] != normal)[:limit]
        a_texts = [texts[i] for i in test_idx[attack]]
        a_labels = [classes[c] for c in first["y"][test_idx[attack]]]
        p_clean_all = first[f"test_probs_{k}"][attack]
        # 저장 가중치가 cvprobs 를 만든 그 가중치인지 원문 재추론으로 확인한다.
        max_drift = max(max_drift, float(np.abs(predict(net, a_texts[:256], device) - p_clean_all[:256]).max()))
        for name in names:
            rows, mutated = mutate_rows(a_texts, a_labels, name, SEED)
            if not len(rows):
                continue
            p_clean, p_mut = p_clean_all[rows], predict(net, mutated, device)
            t = tally[name]
            t["n"] += len(rows)
            t["attack_clean"] += int((p_clean.argmax(1) != normal).sum())
            t["attack_mut"] += int((p_mut.argmax(1) != normal).sum())
            for g in GATES:
                t["clean"][g] += int(gate_mask(point[g], p_clean, p_clean.max(1)).sum())
                t["mut"][g] += int(gate_mask(point[g], p_mut, p_mut.max(1)).sum())
        print(f"fold {k} 완료", flush=True)
    if max_drift > 1e-3:
        raise ValueError(f"저장 가중치 재추론이 cvprobs 와 다릅니다(최대 차 {max_drift:.2e}).")

    per_mutation = {}
    for name, t in tally.items():
        if not t["n"]:
            per_mutation[name] = {"n_changed": 0}
            continue
        r = {s: {g: t[s][g] / t["n"] for g in GATES} for s in ("clean", "mut")}
        per_mutation[name] = {
            "n_changed": t["n"], "r_A_clean": r["clean"], "r_A_mutated": r["mut"],
            "inflation": {g: r["mut"][g] / r["clean"][g] if r["clean"][g] > 0 else None for g in GATES},
            # 공격 요청 하나의 기대 지연 = 1차 전량 + 넘긴 몫의 DistilBERT.
            "attack_latency_ms": {s: {g: t2 + r[s][g] * t_r for g in GATES} for s in ("clean", "mut")},
            "stage1_attack_rate": {"clean": t["attack_clean"] / t["n"], "mutated": t["attack_mut"] / t["n"]}}
        print(f"{name}: n={t['n']} " + " ".join(
            f"{g} r_A {r['clean'][g]:.3f}→{r['mut'][g]:.3f}" for g in GATES), flush=True)
    judged = {n: v for n, v in per_mutation.items() if v["n_changed"] >= MIN_ROWS
              and all(v["inflation"][g] is not None for g in GATES)}
    mean_inflation = {g: float(np.mean([v["inflation"][g] for v in judged.values()])) if judged else None
                      for g in GATES}
    ratio = mean_inflation["g2"] / mean_inflation["b0"] if judged else None
    return {"prereg": "docs/paper_revision/03_prereg_v2.md §E15", "seed": SEED, "ckpt_tag": CKPT_TAG,
            "latency_ms": {"t2": t2, "t_R": t_r}, "classes": classes, "operating_points": points,
            "pooled_speedup_real_user": row["pooled_speedup"], "reinference_max_abs_diff": max_drift,
            "limit": limit, "per_mutation": per_mutation, "judged_mutations": sorted(judged),
            "mean_inflation": mean_inflation,
            "E15": {"g2_over_b0": ratio,
                    "cost_of_g2_clause": None if ratio is None else bool(ratio >= INFLATION_RATIO_LIMIT)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="fold 마다 공격 행 수 제한(스모크, 저장 안 함)")
    args = parser.parse_args()
    out = GF.RESULTS / "dl_a_slowdown_srbh_4class.json"
    try:
        result = analyze(args.limit)
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] E15 측정 실패: {error}", file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ("mean_inflation", "E15", "judged_mutations")},
                     ensure_ascii=False, indent=1))
    if not args.limit:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"저장: {out.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
