"""E10 — DL-A: DistilBERT 관문 · 같은 조건 유사 연구(WAMM'25 DistilBERT) 재구현 대비 · char-CNN → DistilBERT 판정.

판정 규칙은 `docs/paper_revision/03_prereg_v2.md` §E10 을 그대로 옮긴 것이다(학습 전 확정, 커밋 00c52ce).

  --latency : t2(char-CNN)·t_front·t_rest(DL-C)·t_R(토큰화 + DistilBERT, 실제 길이)를 같은 실행 안에서
              요청마다 번갈아 잰다(CPU·배치 1·스레드 1·p50, E8 과 같은 표본). E8 지연 파일은 건드리지 않는다.
              지연은 가중치 값과 무관하므로 학습하지 않은 같은 구조의 모델로 잰다.
  --analyze : E10-1(관문)·E10-2(재구현 R 대비)·E10-3(캐스케이드 char-CNN → DistilBERT)을 판정한다.
              `--seed 43` 은 E11 확인(§E11) — 지연은 E10 측정값을 그대로 쓰고, 판정은 합산 배수(조화평균)로 한다.
  E12(§E12, 원 설정 128 토큰판): `--latency --max-tokens 512,128` 은 두 길이의 t_R 을 같은 실행에서 요청마다
              번갈아 재 별도 파일에 둔다(E10 지연 파일은 그대로). `--analyze --max-tokens 128` 은 그 파일의 값만 쓰고
              E12-1(128 vs 512 단독)과 병기용 "2차 512 캐스케이드 vs R128"을 더한다.
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
FULL_TOKENS = text_models.DISTILBERT_MAX_TOKENS   # E10·E11 판본(주 기준). 이 값이면 파일명에 토큰 축이 없다.
E12_TOKENS = (FULL_TOKENS, 128)   # E12 지연 실행 — E13·E14 도 이 실행 값을 쓴다(§E13·§E14).
E10_E11_SEEDS = (42, 43)          # E10 지연 실행(45.15ms)으로 판정한 seed. 이후 seed 는 E12 실행 값.


def tok_suffix(max_tokens: int) -> str:
    return "" if max_tokens == FULL_TOKENS else f"_tok{max_tokens}"


def distilbert_cvprobs(seed: int, max_tokens: int = FULL_TOKENS) -> Path:
    # tagging.build_tag 와 같은 규칙(토큰 축은 lr 뒤, 봉인 앞).
    return GF.RESULTS / (f"cvprobs_srbh_4class_distilbert_raw_lr2e-05{tok_suffix(max_tokens)}"
                         f"_sealed{DC.cvprobs_suffix(seed)}.npz")


def latency_path(tokens: tuple[int, ...], threads: int = 1) -> Path:
    """E10 은 512 만 잰 기존 파일, 그 밖의 조합은 별도 파일 — 실행이 다른 지연 값을 섞지 않으려고 파일째 가른다.
    스레드 수가 1 이 아니면(B3) `_thr{n}` 을 붙인다 — 스레드 1 판정 값을 덮어쓰지 않는다."""
    thr = "" if threads == 1 else f"_thr{threads}"
    if tokens == (FULL_TOKENS,):
        return LATENCY_JSON.with_name(f"{LATENCY_JSON.stem}{thr}.json")
    return LATENCY_JSON.with_name(f"{LATENCY_JSON.stem}_tok{'-'.join(map(str, tokens))}{thr}.json")


def analysis_tokens(seed: int, max_tokens: int) -> tuple[int, ...]:
    """어느 지연 실행을 쓸지 — E10·E11(512, seed 42·43)만 E10 실행, 나머지는 E12 실행(§E12·§E13)."""
    if max_tokens == FULL_TOKENS and seed in E10_E11_SEEDS:
        return (FULL_TOKENS,)
    return E12_TOKENS


def t_r_key(max_tokens: int) -> str:
    return "t_R" + tok_suffix(max_tokens)


@torch.inference_mode()
def measure_latency(texts: list[str], repeats: int, warmup: int = 20,
                    tokens: tuple[int, ...] = (FULL_TOKENS,), threads: int = 1) -> dict:
    from transformers import AutoTokenizer

    torch.set_num_threads(threads)
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
        bert_times = {}
        # 토큰 길이가 여럿이면 요청마다 순서를 뒤집는다 — 늘 뒤에 재는 쪽만 캐시 덕을 보는 치우침을 없앤다.
        for n in (tokens if flip[0] else tokens[::-1]):
            t3 = clock()
            # 패딩 없는 실제 길이 — 배치 1 이라 마스크는 전부 1 이고 모델 안의 자르기도 일어나지 않는다.
            ids = tok(text, truncation=True, max_length=n, return_tensors="pt")
            t4 = clock()
            bert(ids["input_ids"])
            bert_times[n] = (t4 - t3, clock() - t4)
        flip[0] = not flip[0]
        row = [t1 - t0, t2 - t1, emb + conv3 + aux3, conv5 + conv7 + head]
        for n in tokens:
            row += list(bert_times[n])
        return np.array(row, dtype=np.int64)

    flip = [True]
    for text in texts[:warmup]:
        one(text)
    names = ("enc", "full_fwd", "ee_front_fwd", "ee_rest_fwd")
    for n in tokens:
        names += (f"tokenize{tok_suffix(n)}", f"bert_fwd{tok_suffix(n)}")
    times = np.empty((repeats, len(texts), len(names)), dtype=np.int64)
    for r in range(repeats):
        for i, text in enumerate(texts):
            times[r, i] = one(text)
        print(f"  반복 {r + 1}/{repeats}", flush=True)
    parts = {name: LF.stats(times[:, :, k].ravel() / 1e6) for k, name in enumerate(names)}
    p50 = {name: row["p50"] for name, row in parts.items()}
    ms = {"t2": p50["enc"] + p50["full_fwd"], "t_front": p50["enc"] + p50["ee_front_fwd"],
          "t_rest": p50["ee_rest_fwd"]}
    for n in tokens:
        ms[t_r_key(n)] = p50[f"tokenize{tok_suffix(n)}"] + p50[f"bert_fwd{tok_suffix(n)}"]
    return {"threads": threads, "tokens": list(tokens), "components": parts, "ms": ms}


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


def pooled_speedup(per_fold) -> float:
    """평균 지연 기준 배수 — fold 운영점 E_π 의 평균으로 S 를 다시 낸 값.

    S_k = t_R / (t2 + E_k·t_R) 라 1/S_k 가 E_k 에 선형이다 → 평균 E 로 낸 S 는 fold S 의 조화평균과 같다.
    산술평균은 넘김이 많은 fold 하나(E10 seed 42 의 fold 4, 2.95배)를 거의 무시해 배포 비용을 과대평가한다(§E11).
    """
    return float(1.0 / np.mean(1.0 / np.asarray(per_fold, dtype=float)))


def judge_pooled(row: dict, f1_r: float) -> dict:
    """E11: E10-3 의 세 조건에서 조건 1 의 평균만 합산 배수로 바꾼 판정."""
    folds = np.asarray(row["per_fold_speedup"]["b0"])
    checks = {"pooled_speedup_ge_1.2": bool(pooled_speedup(folds) >= DC.MIN_SPEEDUP),
              "all_folds_gt_1": bool((folds > 1).all()),
              "f1_loss_le_0.3pp": bool(row["b0"]["macro_f1"] >= f1_r - DC.MAX_F1_LOSS)}
    return {**checks, "above": all(checks.values()), "pooled_speedup": pooled_speedup(folds),
            "f1_vs_r_pp": (row["b0"]["macro_f1"] - f1_r) * 100}


def above_r(t_r: float, t_x: float, f1_x, f1_r) -> dict:
    """E10-2: t_R / t_X ≥ 1.2 그리고 F1_X ≥ F1_R − 0.3%p."""
    checks = {"speed_ratio_ge_1.2": bool(t_r / t_x >= DC.MIN_SPEEDUP),
              "f1_loss_le_0.3pp": bool(np.mean(f1_x) >= np.mean(f1_r) - DC.MAX_F1_LOSS)}
    return {"t_x_ms": t_x, "speed_ratio": t_r / t_x, "macro_f1": float(np.mean(f1_x)), **checks,
            "above": all(checks.values()), **versus(f1_x, f1_r)}


def cascade_block(solo: dict, bert: dict, strata, t2: float, t_r: float, f1_solo, f1_r) -> dict:
    """E10-3: 1차 char-CNN → 2차 DistilBERT. 배수의 기준은 2차 단독 R 이다."""
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
            # 평균 비용 기준(§E11) — 산술평균은 넘김이 많은 fold 의 지연을 거의 무시한다.
            latency_ratio_pooled=pooled_speedup(row["per_fold_speedup"]["b0"]) * t2 / t_r,
            more_accurate=bool(vs_charcnn["mean_diff_pp"] > 0 and p is not None and p < 0.05))
        row["pooled_speedup"] = {gate: pooled_speedup(s) for gate, s in row["per_fold_speedup"].items()}
        block[view] = {**row, "e8_vs_r": DC.judge(row, float(np.mean(f1_r))),
                       "e11_vs_r": judge_pooled(row, float(np.mean(f1_r))), "vs_charcnn": vs_charcnn}
    return block


def analyze(seed: int = DC.BASE_SEED, max_tokens: int = FULL_TOKENS) -> dict:
    # 128 판본은 512 와 같은 실행에서 잰 지연만 쓴다(§E12) — E10 의 45.15ms 와 섞지 않는다.
    tokens = analysis_tokens(seed, max_tokens)
    ms = json.loads(latency_path(tokens).read_text(encoding="utf-8"))["latency"]["ms"]
    t2, t_front, t_rest, t_r = ms["t2"], ms["t_front"], ms["t_rest"], ms[t_r_key(max_tokens)]
    solo = DC.load_cvprobs("charcnn", seed=seed)
    bert = load_npz(distilbert_cvprobs(seed, max_tokens))
    # DL-C(charcnn_ee)는 seed 42·43 만 학습했다 → seed 44·45(E13)에서는 E10-2 의 DL-C 행을 건너뛴다.
    has_ee = (GF.RESULTS / DC.CVPROBS.format("charcnn_ee", DC.cvprobs_suffix(seed, aux=True))).exists()
    if has_ee:
        ee_main = DC.load_cvprobs("charcnn_ee", seed=seed)
        ee_aux = DC.load_cvprobs("charcnn_ee", aux=True, seed=seed)
        DC.check_aligned(solo, bert, ee_main, ee_aux)
    else:
        DC.check_aligned(solo, bert)
    strata = np.asarray(cross_validate.load_pool_strata("srbh_4class", exclude_test=True))
    f1_solo, f1_r = DC.fold_test_f1(solo), DC.fold_test_f1(bert)
    prereg = ("§E12" if max_tokens != FULL_TOKENS else "§E10" if seed == DC.BASE_SEED
              else "§E11" if seed in E10_E11_SEEDS else "§E13")
    result = {"prereg": f"docs/paper_revision/03_prereg_v2.md {prereg}", "seed": seed, "pi": GF.PI,
              "max_tokens": max_tokens,
              "latency_ms": ms, "charcnn_per_fold_macro_f1": f1_solo, "distilbert_per_fold_macro_f1": f1_r,
              "charcnn_macro_f1": float(np.mean(f1_solo)), "distilbert_macro_f1": float(np.mean(f1_r))}

    # E10-1 관문: 진입 조건이지 우열 주장이 아니다(보정 t 는 병기).
    result["e10_1"] = {"pass": bool(np.mean(f1_r) > np.mean(f1_solo)), **versus(f1_r, f1_solo)}

    # E10-2: X = char-CNN 단독, DL-C(같은 seed 의 운영점 — 이 실행의 t_front·t_rest 로 다시 낸다).
    e10_2 = {"charcnn": above_r(t_r, t2, f1_solo, f1_r)}
    paths = GF.fold_paths(ee_aux, ee_main, strata)[0] if has_ee else None
    for view, col in GC.VIEWS:
        if paths is None:
            e10_2[f"dl_c_{view}"] = {"above": False, "reason": f"charcnn_ee seed {seed} 미학습"}
            continue
        row = GC.secondary_rule(paths, fold_val_f1(solo), col, t_front, t_rest)
        if "reason" in row:
            e10_2[f"dl_c_{view}"] = {"above": False, "reason": row["reason"]}
            continue
        # secondary_rule 의 fold 배수 = t_rest / (t_front + E_π·t_rest) → fold 별 DL-C 지연으로 되돌린다.
        t_x = float(np.mean([t_rest / s for s in row["per_fold_speedup"]["b0"]]))
        e10_2[f"dl_c_{view}"] = above_r(t_r, t_x, row["per_fold_macro_f1"]["b0"], f1_r)
    result["e10_2"] = e10_2

    # E10-3 (E12 에서는 E12-3 — 판정은 e11_vs_r, 기준 R 은 같은 토큰 길이의 단독).
    result["e10_3"] = cascade_block(solo, bert, strata, t2, t_r, f1_solo, f1_r)
    if max_tokens == FULL_TOKENS:
        return result

    # E12-1 (판정): 같은 seed 의 512 판본과 fold 대응 비교. 두 seed 를 묶는 판정은 문서에서 한다.
    bert_full = load_npz(distilbert_cvprobs(seed))
    DC.check_aligned(solo, bert_full)
    f1_full = DC.fold_test_f1(bert_full)
    result["e12_1"] = {"distilbert_full_macro_f1": float(np.mean(f1_full)), **versus(f1_r, f1_full)}
    # 병기(판정 아님): 2차는 512 그대로, 기준만 R128 — 토큰 길이가 다른 비교라 헤드라인 금지.
    t_full = ms[t_r_key(FULL_TOKENS)]
    row = cascade_block(solo, bert_full, strata, t2, t_full, f1_solo, f1_full)["real_user"]
    result["bystander_cascade_full_vs_r"] = {"reason": row["reason"]} if "reason" in row else {
        # 배수 기준만 t_R512 → t_R128 로 옮긴다: S' = S · t_R128 / t_R512.
        "pooled_speedup_vs_r": pooled_speedup(row["per_fold_speedup"]["b0"]) * t_r / t_full,
        "f1_vs_r_pp": (row["b0"]["macro_f1"] - float(np.mean(f1_r))) * 100}
    return result


def report(result: dict) -> None:
    ms = result["latency_ms"]
    print(f"지연(ms) t2 {ms['t2']:.3f} · t_front {ms['t_front']:.3f} · t_rest {ms['t_rest']:.3f} · t_R {ms['t_R']:.3f}")
    print(f"F1 char-CNN {result['charcnn_macro_f1']:.4f} · DistilBERT({result.get('max_tokens', FULL_TOKENS)} 토큰)"
          f" {result['distilbert_macro_f1']:.4f}")
    if "e12_1" in result:
        g = result["e12_1"]
        print(f"[E12-1] {result['max_tokens']} − {FULL_TOKENS} 토큰 {g['mean_diff_pp']:+.2f}%p"
              f" (fold {np.round(g['diff_pp'], 2).tolist()}) · 보정 t {g['corrected']}")
        print(f"[병기] 2차 {FULL_TOKENS} 캐스케이드 vs R{result['max_tokens']}: {result['bystander_cascade_full_vs_r']}")
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
        pooled = row["pooled_speedup"]
        print(f"  {view}: B0 합산 {pooled['b0']:.3f}배 · 산술 {row['b0']['speedup']:.3f}배"
              f" (fold {np.round(row['per_fold_speedup']['b0'], 3).tolist()})"
              f" F1 {row['b0']['macro_f1']:.4f} r_N {row['b0']['r_N']:.3f} r_A {row['b0']['r_A']:.3f}"
              f" | G2 합산 {pooled['g2']:.3f}배 · 산술 {row['g2']['speedup']:.3f}배")
        print(f"    R 대비 E10-3(산술) {row['e8_vs_r']}")
        print(f"    R 대비 E11(합산) {row['e11_vs_r']}")
        print(f"    char-CNN 대비 F1 {vc['mean_diff_pp']:+.2f}%p (보정 p {vc['corrected']['p']}) · 지연 배율 합산"
              f" {vc['latency_ratio_pooled']:.3f} · 산술 {vc['latency_ratio']:.3f} · 더 정확 {vc['more_accurate']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--latency", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="--latency 를 n=30·1회로, 저장 없이")
    parser.add_argument("--seed", type=int, default=DC.BASE_SEED, help="--analyze 의 CV seed (43 = E11 확인)")
    parser.add_argument("--threads", type=int, default=1, help="--latency 의 torch 스레드 수(B3: 4·8)")
    parser.add_argument("--max-tokens", default=str(FULL_TOKENS),
                        help="DistilBERT 토큰 길이. --latency 는 쉼표 목록(512,128 = E12), --analyze 는 하나")
    args = parser.parse_args()
    try:
        tokens = tuple(int(n) for n in args.max_tokens.split(","))
    except ValueError:
        parser.error(f"--max-tokens 는 정수(쉼표 목록)여야 합니다: {args.max_tokens}")
    if args.analyze and len(tokens) != 1:
        parser.error("--analyze 의 --max-tokens 는 하나만 받습니다.")
    try:
        if args.latency:
            n, repeats = (30, 1) if args.smoke else (args.n, args.repeats)
            texts_all, _ = LF.data_text.load_text_split(LF.TRACK, "val", LF.TEXT)
            picked = np.random.default_rng(LF.SEED).choice(len(texts_all), n, replace=False)
            result = {"env": LF.environment(args), "n": n, "repeats": repeats,
                      "latency": measure_latency([texts_all[i] for i in picked], repeats, tokens=tokens,
                                                         threads=args.threads)}
            print(json.dumps(result["latency"]["ms"], indent=2))
            if not args.smoke:
                path = latency_path(tokens, args.threads)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"저장: {path.relative_to(PROJECT_ROOT).as_posix()}")
            return 0
        result = analyze(args.seed, tokens[0])
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] E10 실패: {error}", file=sys.stderr)
        return 1
    output = OUTPUT.with_name(f"{OUTPUT.stem}{tok_suffix(tokens[0])}{DC.cvprobs_suffix(args.seed)}.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"seed {args.seed}")
    report(result)
    print(f"저장: {output.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
