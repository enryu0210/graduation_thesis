"""G5(docs/14 §8.14) — 표면 변형이 공격 행의 넘김률을 얼마나 부풀리는지 게이트별로 잰다(비용 기반 공격면).

`run_evasion.py`·`mutations.py` 는 XSS 라벨 체계에 묶여 있어 건드리지 않고, 변형 함수만 가져와 SR-BH 라벨에 매핑한다:
SQLInjection → 공통+SQLi 전용, CommandInjection → 공통+CmdI 전용, CodeInjection → 공통만.
운영점(τ)은 clean val 에서 고른 값(gate_g6 고정 split 결과)을 그대로 쓰고 재튜닝하지 않는다.
⚠️ 변형을 F2 원문 전체(URI+body)에 적용하므로 일부 변형은 HTTP 의미를 보존하지 않는다(problem-space 한계).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.attacks import mutations as M
from src.eval.gate_g4_e1a import build_tensors
from src.eval.gate_g6_cv import gate_mask
from src.models import cascade

GATES = ("b0", "g2")   # H-G6 채택 점수 없음 → 채택 게이트 자리는 G2(§8.15)
LABEL_MUTATIONS = {"SQLInjection": M.CLASS_MUTATIONS["SQLInjection"],
                   "CommandInjection": M.CLASS_MUTATIONS["CommandInjection"],
                   "CodeInjection": list(M.COMMON)}
INFLATION_RATIO_LIMIT = 1.5


def mutate_rows(texts, labels, name, seed):
    """그 변형이 허용된 클래스의 행에만 적용하고, 원문이 바뀐 행만 남긴다."""
    rng = random.Random(seed)
    rows, mutated = [], []
    for i, (text, label) in enumerate(zip(texts, labels)):
        if name not in LABEL_MUTATIONS.get(label, ()):
            continue
        changed = M.apply_mutation(name, text, rng)
        if changed != text:
            rows.append(i)
            mutated.append(changed)
    return np.array(rows, dtype=int), mutated


def gate_stats(points, p1, p2, normal):
    conf = p1.max(1)
    out = {}
    for g in GATES:
        mask = gate_mask(points[g], p1, conf)
        pred = np.where(mask, p2.argmax(1), p1.argmax(1))
        out[g] = {"r_A": float(mask.mean()), "attack_rate": float((pred != normal).mean())}
    return out


def analyze(args):
    import data_text
    g6 = json.loads(args.g6_json.read_text(encoding="utf-8"))
    classes = g6["classes"]
    normal = classes.index("Normal")
    points = {g: g6["gates"][g]["operating_point"] for g in GATES}
    with np.load(args.probs, allow_pickle=False) as archive:
        p1_test, p2_test, y_test = archive["p1_test"], archive["p2_test"], archive["y_test"]
    texts, labels = data_text.load_text_split("srbh_4class", "test", "raw")
    if not np.array_equal(np.asarray(data_text.encode_labels_with(labels, classes)), y_test):
        raise ValueError("test CSV 라벨이 확률 파일과 다릅니다(행 순서 불일치).")
    attack = np.flatnonzero(y_test != normal)
    if args.limit:
        attack = attack[:args.limit]
    a_texts, a_labels = [texts[i] for i in attack], [labels[i] for i in attack]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = cascade.load_net("cnn", len(classes), "srbh_4class", "raw", False, device, "rgb", 3,
                            ("raw_byte", "char_class", "local_entropy"))
    net2 = cascade.load_net("charcnn", len(classes), "srbh_4class", "raw", True, device)
    names = list(dict.fromkeys(m for ms in LABEL_MUTATIONS.values() for m in ms))
    per_mutation = {}
    for name in names:
        rows, mutated = mutate_rows(a_texts, a_labels, name, args.seed)
        if not len(rows):
            per_mutation[name] = {"n_changed": 0}
            continue
        x_img, x_seq = build_tensors(mutated)
        mut = gate_stats(points, cascade.predict_probs(net1, x_img, device, 512),
                         cascade.predict_probs(net2, x_seq, device, 512), normal)
        clean = gate_stats(points, p1_test[attack[rows]], p2_test[attack[rows]], normal)
        per_mutation[name] = {"n_changed": int(len(rows)), "clean": clean, "mutated": mut,
                              "inflation": {g: (mut[g]["r_A"] / clean[g]["r_A"]) if clean[g]["r_A"] > 0 else None
                                            for g in GATES}}
        print(f"{name}: n={len(rows)} " + " ".join(
            f"{g} r_A {clean[g]['r_A']:.3f}→{mut[g]['r_A']:.3f}" for g in GATES), flush=True)
    mean_inflation = {g: float(np.mean([v["inflation"][g] for v in per_mutation.values()
                                        if v.get("inflation", {}).get(g) is not None])) for g in GATES}
    ratio = mean_inflation["g2"] / mean_inflation["b0"]
    # 사전 정의는 변형별 배율의 평균이지만, clean r_A 가 0 에 가까운 소표본 변형이 평균을 흔든다 →
    # 행 수 가중 합산 배율(Σ넘김_mut / Σ넘김_clean)을 보조로 함께 남긴다(판정은 평균 기준 그대로).
    done = [v for v in per_mutation.values() if v["n_changed"]]
    pooled = {g: sum(v["mutated"][g]["r_A"] * v["n_changed"] for v in done)
              / max(sum(v["clean"][g]["r_A"] * v["n_changed"] for v in done), 1e-12) for g in GATES}
    return {"probs": args.probs.name, "g6_json": args.g6_json.name, "classes": classes,
            "label_mutations": LABEL_MUTATIONS, "n_attack_rows": int(len(attack)),
            "per_mutation": per_mutation, "mean_inflation": mean_inflation, "pooled_inflation": pooled,
            "H-G5": {"g2_over_b0": ratio, "cost_of_gate_improvement": bool(ratio >= INFLATION_RATIO_LIMIT)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=Path, required=True, help="T1 _s2bal tradeoff_probs npz(clean 확률)")
    parser.add_argument("--g6-json", type=Path, required=True, help="gate_g6.py 고정 split 결과(운영점 τ 출처)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="스모크용 공격 행 수 제한(저장하지 않는다)")
    args = parser.parse_args()
    out = PROJECT_ROOT / "experiments/results/gate_g5_mutation_srbh_4class.json"
    try:
        if not args.limit and out.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {out.name}")
        result = analyze(args)
        if not args.limit:
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
        print(json.dumps({"mean_inflation": result["mean_inflation"], "pooled_inflation": result["pooled_inflation"], "H-G5": result["H-G5"]},
                         ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"G5 변형 넘김 측정 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
