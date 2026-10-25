"""E10 — DL-A: DistilBERT 관문 · 같은 조건 유사 연구(WAMM'25 DistilBERT) 재구현 대비 · char-CNN → DistilBERT 판정.

판정 규칙은 `docs/paper_revision/03_prereg_v2.md` §E10 을 그대로 옮긴 것이다(학습 전 확정, 커밋 00c52ce).

  --latency : t2(char-CNN)·t_front·t_rest(DL-C)·t_R(토큰화 + DistilBERT, 실제 길이)를 같은 실행 안에서
              요청마다 번갈아 잰다(CPU·배치 1·스레드 1·p50, E8 과 같은 표본). E8 지연 파일은 건드리지 않는다.
              지연은 가중치 값과 무관하므로 학습하지 않은 같은 구조의 모델로 잰다.
  --analyze : E10-1(관문)·E10-2(재구현 R 대비)·E10-3(캐스케이드 char-CNN → DistilBERT)을 판정한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import latency_fair as LF  # noqa: E402  (data_text·text_models 경로도 여기서 잡힌다)
from src.eval import cross_validate  # noqa: E402
from src.eval import dl_cascade as DC  # noqa: E402
from src.eval import gate_confirm as GC  # noqa: E402
from src.eval import gate_frontier as GF  # noqa: E402
from src.models import gate_signals as gates  # noqa: E402
from src.models import text_models  # noqa: E402

LATENCY_JSON = GF.RESULTS / "dl_a_latency_srbh_4class.json"
OUTPUT = GF.RESULTS / "dl_a_srbh_4class.json"
DISTILBERT_CVPROBS = GF.RESULTS / "cvprobs_srbh_4class_distilbert_raw_lr2e-05_sealed.npz"


@torch.inference_mode()
def measure_latency(texts: list[str], repeats: int, warmup: int = 20) -> dict:
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    clock = time.perf_counter_ns
    torch.manual_seed(LF.SEED)
    full = text_models.build_model("charcnn", DC.N_CLASSES).eval()
    ee = text_models.build_model("charcnn_ee", DC.N_CLASSES).eval()
    bert = text_models.build_model("distilbert", DC.N_CLASSES).eval()
    tok = AutoTokenizer.from_pretrained(text_models.DISTILBERT_NAME)

    def one(text: str) -> np.ndarray:
        t0 = clock()
        x = LF.enc_seq(text, False)
        t1 = clock()
        full(x)
        t2 = clock()
        _, (emb, conv3, conv5, conv7, head, aux3) = LF.charcnn_parts(ee, ee.aux_head, x, clock)
        t3 = clock()
        # 패딩 없는 실제 길이 — 배치 1 이라 마스크는 전부 1 이고 모델 안의 자르기도 일어나지 않는다.
        ids = tok(text, truncation=True, max_length=text_models.DISTILBERT_MAX_TOKENS, return_tensors="pt")
        t4 = clock()
        bert(ids["input_ids"])
        t5 = clock()
        return np.array((t1 - t0, t2 - t1, emb + conv3 + aux3, conv5 + conv7 + head, t4 - t3, t5 - t4),
                        dtype=np.int64)

    for text in texts[:warmup]:
        one(text)
    names = ("enc", "full_fwd", "ee_front_fwd", "ee_rest_fwd", "tokenize", "bert_fwd")
    times = np.empty((repeats, len(texts), len(names)), dtype=np.int64)
    for r in range(repeats):
        for i, text in enumerate(texts):
            times[r, i] = one(text)
        print(f"  반복 {r + 1}/{repeats}", flush=True)
    parts = {name: LF.stats(times[:, :, k].ravel() / 1e6) for k, name in enumerate(names)}
    p50 = {name: row["p50"] for name, row in parts.items()}
    return {"threads": 1, "components": parts,
            "ms": {"t2": p50["enc"] + p50["full_fwd"], "t_front": p50["enc"] + p50["ee_front_fwd"],
                   "t_rest": p50["ee_rest_fwd"], "t_R": p50["tokenize"] + p50["bert_fwd"]}}


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def fold_val_f1(probs: dict) -> list[float]:
    return [gates._macro_f1_fast(probs["y"][probs[f"val_idx_{k}"]], probs[f"val_probs_{k}"].argmax(1), DC.N_CLASSES)
            for k in range(1, 6)]


def versus(f1_x, f1_r) -> dict:
    diffs = np.subtract(f1_x, f1_r)
    return {"diff_pp": (diffs * 100).tolist(), "mean_diff_pp": float(diffs.mean() * 100),
            "corrected": GC.corrected_t(diffs)}


def above_r(t_r: float, t_x: float, f1_x, f1_r) -> dict:
    """E10-2: t_R / t_X ≥ 1.2 그리고 F1_X ≥ F1_R − 0.3%p."""
    checks = {"speed_ratio_ge_1.2": bool(t_r / t_x >= DC.MIN_SPEEDUP),
              "f1_loss_le_0.3pp": bool(np.mean(f1_x) >= np.mean(f1_r) - DC.MAX_F1_LOSS)}
    return {"t_x_ms": t_x, "speed_ratio": t_r / t_x, "macro_f1": float(np.mean(f1_x)), **checks,
            "above": all(checks.values()), **versus(f1_x, f1_r)}


def analyze() -> dict:
    ms = json.loads(LATENCY_JSON.read_text(encoding="utf-8"))["latency"]["ms"]
    t2, t_front, t_rest, t_r = ms["t2"], ms["t_front"], ms["t_rest"], ms["t_R"]
    solo, bert = DC.load_cvprobs("charcnn"), load_npz(DISTILBERT_CVPROBS)
    ee_main, ee_aux = DC.load_cvprobs("charcnn_ee"), DC.load_cvprobs("charcnn_ee", aux=True)
    DC.check_aligned(solo, bert, ee_main, ee_aux)
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    f1_solo, f1_r = DC.fold_test_f1(solo), DC.fold_test_f1(bert)
    result = {"prereg": "docs/paper_revision/03_prereg_v2.md §E10", "seed": DC.BASE_SEED, "pi": GF.PI,
              "latency_ms": ms, "charcnn_per_fold_macro_f1": f1_solo, "distilbert_per_fold_macro_f1": f1_r,
              "charcnn_macro_f1": float(np.mean(f1_solo)), "distilbert_macro_f1": float(np.mean(f1_r))}

    # E10-1 관문: 진입 조건이지 우열 주장이 아니다(보정 t 는 병기).
    result["e10_1"] = {"pass": bool(np.mean(f1_r) > np.mean(f1_solo)), **versus(f1_r, f1_solo)}

    # E10-2: X = char-CNN 단독, DL-C(seed 42 운영점 — 이 실행의 t_front·t_rest 로 다시 낸다).
    e10_2 = {"charcnn": above_r(t_r, t2, f1_solo, f1_r)}
    paths, _ = GF.fold_paths(ee_aux, ee_main, strata)
    for view, col in GC.VIEWS:
        row = GC.secondary_rule(paths, fold_val_f1(solo), col, t_front, t_rest)
        if "reason" in row:
            e10_2[f"dl_c_{view}"] = {"above": False, "reason": row["reason"]}
            continue
        # secondary_rule 의 fold 배수 = t_rest / (t_front + E_π·t_rest) → fold 별 DL-C 지연으로 되돌린다.
        t_x = float(np.mean([t_rest / s for s in row["per_fold_speedup"]["b0"]]))
        e10_2[f"dl_c_{view}"] = above_r(t_r, t_x, row["per_fold_macro_f1"]["b0"], f1_r)
    result["e10_2"] = e10_2

    # E10-3: 1차 char-CNN → 2차 DistilBERT. 배수의 기준은 2차 단독 R 이다.
    paths, (solo1, solo2) = GF.fold_paths(solo, bert, strata)
    block = {"stage1_solo_macro_f1": float(solo1), "stage2_solo_macro_f1": float(solo2), "upper_bound": t_r / t2}
    for view, col in GC.VIEWS:
        row = GC.secondary_rule(paths, fold_val_f1(bert), col, t2, t_r)
        if "reason" in row:
            block[view] = {**row, "e8_vs_r": {"above": False, "reason": row["reason"]}}
            continue
        row.pop("adopted")   # C-G2 의 G2 채택 규칙이라 E10 과 무관하다.
        f1_cascade = row["per_fold_macro_f1"]["b0"]
        vs_charcnn = versus(f1_cascade, f1_solo)
        p = vs_charcnn["corrected"]["p"]
        vs_charcnn.update(
            # 모든 요청이 char-CNN 을 거치므로 char-CNN 단독 대비 배율은 항상 1 미만이다.
            latency_ratio=float(np.mean([s * t2 / t_r for s in row["per_fold_speedup"]["b0"]])),
            more_accurate=bool(vs_charcnn["mean_diff_pp"] > 0 and p is not None and p < 0.05))
        block[view] = {**row, "e8_vs_r": DC.judge(row, float(np.mean(f1_r))), "vs_charcnn": vs_charcnn}
    result["e10_3"] = block
    return result


def report(result: dict) -> None:
    ms = result["latency_ms"]
    print(f"지연(ms) t2 {ms['t2']:.3f} · t_front {ms['t_front']:.3f} · t_rest {ms['t_rest']:.3f} · t_R {ms['t_R']:.3f}")
    print(f"F1 char-CNN {result['charcnn_macro_f1']:.4f} · DistilBERT {result['distilbert_macro_f1']:.4f}")
    g = result["e10_1"]
    print(f"\n[E10-1 관문] {'통과' if g['pass'] else '미통과'} · 차이 {np.round(g['diff_pp'], 2).tolist()}%p"
          f" · 보정 t {g['corrected']}")
    print("\n[E10-2 재구현 R 대비]")
    for name, row in result["e10_2"].items():
        if "reason" in row:
            print(f"  {name}: {row['reason']}")
            continue
        print(f"  {name}: t_X {row['t_x_ms']:.3f}ms · t_R/t_X {row['speed_ratio']:.2f}배 · F1 {row['macro_f1']:.4f}"
              f" (R 대비 {row['mean_diff_pp']:+.2f}%p, 보정 p {row['corrected']['p']}) → 위 {row['above']}")
    block = result["e10_3"]
    print(f"\n[E10-3 char-CNN → DistilBERT] 상한 {block['upper_bound']:.2f}배")
    for view, _ in GC.VIEWS:
        row = block[view]
        if "reason" in row:
            print(f"  {view}: {row['reason']}")
            continue
        vc = row["vs_charcnn"]
        print(f"  {view}: B0 {row['b0']['speedup']:.3f}배 (fold {np.round(row['per_fold_speedup']['b0'], 3).tolist()})"
              f" F1 {row['b0']['macro_f1']:.4f} r_N {row['b0']['r_N']:.3f} r_A {row['b0']['r_A']:.3f}"
              f" | G2 {row['g2']['speedup']:.3f}배 | R 대비 {row['e8_vs_r']}")
        print(f"    char-CNN 대비 F1 {vc['mean_diff_pp']:+.2f}%p (보정 p {vc['corrected']['p']}) · 지연 배율"
              f" {vc['latency_ratio']:.3f} · 더 정확 {vc['more_accurate']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--latency", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="--latency 를 n=30·1회로, 저장 없이")
    args = parser.parse_args()
    try:
        if args.latency:
            n, repeats = (30, 1) if args.smoke else (args.n, args.repeats)
            texts_all, _ = LF.data_text.load_text_split(LF.TRACK, "val", LF.TEXT)
            picked = np.random.default_rng(LF.SEED).choice(len(texts_all), n, replace=False)
            result = {"env": LF.environment(args), "n": n, "repeats": repeats,
                      "latency": measure_latency([texts_all[i] for i in picked], repeats)}
            print(json.dumps(result["latency"]["ms"], indent=2))
            if not args.smoke:
                LATENCY_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"저장: {LATENCY_JSON.relative_to(PROJECT_ROOT).as_posix()}")
            return 0
        result = analyze()
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] E10 실패: {error}", file=sys.stderr)
        return 1
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report(result)
    print(f"저장: {OUTPUT.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
