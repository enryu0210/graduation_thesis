"""E8 — DL-B(작은 문자 모델 → char-CNN)·DL-C(char-CNN 내부 조기 종료) 본 측정 판정.

판정 규칙은 `docs/paper_revision/03_prereg_v2.md` §E8 을 그대로 옮긴 것이다(학습 전 확정, 커밋 f5a002a).
비교 대상은 char-CNN 단독(S = 1, `cvprobs_srbh_4class_charcnn_raw_sealed.npz`).

  --latency : t_tiny·t2·t_front·t_rest 를 같은 실행 안에서 번갈아 잰다(CPU·배치 1·스레드 1·p50, E7 과 같은 표본).
              지연은 가중치 값과 무관하므로 학습하지 않은 같은 구조의 모델로 잰다.
  --analyze : 두 후보의 운영점을 fold 마다 val 에서 골라(gate_confirm.secondary_rule) fold-test 배수·F1 로 판정한다.
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
from src.eval import gate_confirm as GC  # noqa: E402
from src.eval import gate_frontier as GF  # noqa: E402
from src.models import gate_signals as gates  # noqa: E402
from src.models import text_models  # noqa: E402

LATENCY_JSON = GF.RESULTS / "dl_cascade_latency_srbh_4class.json"
OUTPUT = GF.RESULTS / "dl_cascade_srbh_4class.json"
CVPROBS = "cvprobs_srbh_4class_{}_raw_sealed{}.npz"
N_CLASSES = 4
# E8 판정 기준(사전 고정).
MIN_SPEEDUP, MAX_F1_LOSS = 1.2, 0.003


@torch.inference_mode()
def measure_latency(texts: list[str], repeats: int, warmup: int = 50) -> dict:
    torch.set_num_threads(1)
    clock = time.perf_counter_ns
    torch.manual_seed(LF.SEED)
    tiny = text_models.build_model("charcnn_tiny", N_CLASSES).eval()
    full = text_models.build_model("charcnn", N_CLASSES).eval()
    ee = text_models.build_model("charcnn_ee", N_CLASSES).eval()

    def one(text: str) -> np.ndarray:
        t0 = clock()
        x = LF.enc_seq(text, False)
        t1 = clock()
        tiny(x)
        t2 = clock()
        full(x)
        t3 = clock()
        _, (emb, conv3, conv5, conv7, head, aux3) = LF.charcnn_parts(ee, ee.aux_head, x, clock)
        return np.array((t1 - t0, t2 - t1, t3 - t2, emb + conv3 + aux3, conv5 + conv7 + head), dtype=np.int64)

    for text in texts[:warmup]:
        one(text)
    times = np.empty((repeats, len(texts), 5), dtype=np.int64)
    for r in range(repeats):
        for i, text in enumerate(texts):
            times[r, i] = one(text)
    parts = {name: LF.stats(times[:, :, k].ravel() / 1e6)
             for k, name in enumerate(("enc", "tiny_fwd", "full_fwd", "ee_front_fwd", "ee_rest_fwd"))}
    p50 = {name: row["p50"] for name, row in parts.items()}
    # 인코딩은 매 요청 한 번 — 1차(또는 앞부분)에만 붙고 넘겨받는 쪽은 다시 하지 않는다.
    return {"threads": 1, "components": parts,
            "ms": {"t_tiny": p50["enc"] + p50["tiny_fwd"], "t2": p50["enc"] + p50["full_fwd"],
                   "t_front": p50["enc"] + p50["ee_front_fwd"], "t_rest": p50["ee_rest_fwd"]}}


def load_cvprobs(model: str, aux: bool = False) -> dict:
    with np.load(GF.RESULTS / CVPROBS.format(model, "_aux" if aux else ""), allow_pickle=False) as data:
        return dict(data)


def check_aligned(reference: dict, *others: dict) -> None:
    """같은 fold·같은 내부 val 을 봤는지 — 어긋나면 넘김 비교가 무의미하다."""
    for other in others:
        if str(other["label_fingerprint"]) != str(reference["label_fingerprint"]):
            raise ValueError("cvprobs 의 라벨 지문이 char-CNN 단독과 다릅니다.")
        for k in range(1, 6):
            if not all(np.array_equal(reference[f"{s}_idx_{k}"], other[f"{s}_idx_{k}"]) for s in ("val", "test")):
                raise ValueError(f"fold {k} 인덱스가 char-CNN 단독과 다릅니다.")


def judge(row: dict, solo_f1: float) -> dict:
    if "reason" in row:
        return {"above": False, "reason": row["reason"]}
    folds = np.array(row["per_fold_speedup"]["b0"])
    checks = {"mean_speedup_ge_1.2": bool(folds.mean() >= MIN_SPEEDUP),
              "all_folds_gt_1": bool((folds > 1).all()),
              "f1_loss_le_0.3pp": bool(row["b0"]["macro_f1"] >= solo_f1 - MAX_F1_LOSS)}
    return {**checks, "above": all(checks.values()), "f1_vs_solo_pp": (row["b0"]["macro_f1"] - solo_f1) * 100}


def analyze() -> dict:
    latency = json.loads(LATENCY_JSON.read_text(encoding="utf-8"))["latency"]["ms"]
    solo = load_cvprobs("charcnn")
    tiny, ee_main, ee_aux = load_cvprobs("charcnn_tiny"), load_cvprobs("charcnn_ee"), load_cvprobs("charcnn_ee", aux=True)
    check_aligned(solo, tiny, ee_main, ee_aux)
    y = solo["y"]
    solo_val = [gates._macro_f1_fast(y[solo[f"val_idx_{k}"]], solo[f"val_probs_{k}"].argmax(1), N_CLASSES)
                for k in range(1, 6)]
    solo_test = float(np.mean([gates._macro_f1_fast(y[solo[f"test_idx_{k}"]], solo[f"test_probs_{k}"].argmax(1),
                                                    N_CLASSES) for k in range(1, 6)]))
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    t2 = latency["t2"]
    # DL-C 는 S = t2 / (t_front + E_π·t_rest): secondary_rule 의 배수 t_rest/(t_front + E_π·t_rest) 에 t2/t_rest 를 곱한다.
    candidates = {"dl_b": (tiny, solo, latency["t_tiny"], t2, 1.0),
                  "dl_c": (ee_aux, ee_main, latency["t_front"], latency["t_rest"], t2 / latency["t_rest"])}
    result = {"prereg": "docs/paper_revision/03_prereg_v2.md §E8", "pi": GF.PI, "latency_ms": latency,
              "charcnn_solo_test_macro_f1": solo_test}
    for name, (first, second, ms1, ms2, scale) in candidates.items():
        paths, (solo1, solo2) = GF.fold_paths(first, second, strata)
        block = {"stage1_solo_macro_f1": float(solo1), "stage2_solo_macro_f1": float(solo2),
                 "upper_bound": scale * ms2 / ms1}
        for view, col in GC.VIEWS:
            row = GC.secondary_rule(paths, solo_val, col, ms1, ms2)
            if "reason" not in row:
                for gate in ("b0", "g2"):
                    row[gate]["speedup"] *= scale
                    row["per_fold_speedup"][gate] = [s * scale for s in row["per_fold_speedup"][gate]]
                row.pop("adopted")   # C-G2 의 G2 채택 규칙이라 E8 과 무관하다.
            block[view] = {**row, "e8": judge(row, solo_test)}
        result[name] = block
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--latency", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="--latency 를 n=50·2회로, 저장 없이")
    args = parser.parse_args()
    try:
        if args.latency:
            n, repeats = (50, 2) if args.smoke else (args.n, args.repeats)
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
        print(f"[오류] E8 실패: {error}", file=sys.stderr)
        return 1
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"char-CNN 단독 test F1 {result['charcnn_solo_test_macro_f1']:.4f} · 지연 {result['latency_ms']}")
    for name in ("dl_b", "dl_c"):
        block = result[name]
        print(f"\n[{name}] 1차 단독 F1 {block['stage1_solo_macro_f1']:.4f} · 2차 단독 F1 {block['stage2_solo_macro_f1']:.4f}"
              f" · 상한 {block['upper_bound']:.2f}배")
        for view, _ in GC.VIEWS:
            row = block[view]
            if "reason" in row:
                print(f"  {view}: {row['reason']}")
                continue
            print(f"  {view}: B0 {row['b0']['speedup']:.3f}배 (fold {np.round(row['per_fold_speedup']['b0'], 3).tolist()})"
                  f" F1 {row['b0']['macro_f1']:.4f} r_N {row['b0']['r_N']:.3f} r_A {row['b0']['r_A']:.3f}"
                  f" | G2 {row['g2']['speedup']:.3f}배 F1 {row['g2']['macro_f1']:.4f} | E8 {row['e8']}")
    print(f"저장: {OUTPUT.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
