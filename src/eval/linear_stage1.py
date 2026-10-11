"""E3 — 1차를 선형 n-gram 모델로 바꾼 캐스케이드 (docs/paper_revision/03_prereg_v2.md §E3)

왜 필요한가:
    논문은 "1차가 이미지, 2차가 문자열"이라는 표현 이질성을 내세운다. 그런데 1차를 값싼 선형 모델로
    바꿔도 같은 속도 이득이 나온다면 이질성은 이득의 필요조건이 아니다. 유사 연구(Tasdemir 등)의
    "선형 1차 → 깊은 2차" 구성을 우리 데이터·우리 2차로 재구현해 같은 그래프에 올리는 것이기도 하다.

단계(지연은 조용한 기기에서 재야 해서 나눴다):
    --fit      fold 를 cross_validate.py 와 똑같이 갈라 1차 후보를 학습하고 확률을 cvprobs 형식으로 저장
    --latency  E1 과 같은 표본·방식(CPU·배치 1·스레드 1, 벡터화 + 예측)으로 요청당 지연 측정
    --analyze  같은 운영점 규칙으로 RGB CNN 1차와 비교(판정), Q 통계와 경계 그림
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval import cross_validate as CV
from src.eval import gate_confirm as GC
from src.eval import gate_frontier as GF
from src.eval.gate_cv import validate_alignment
from src.models import gate_signals as gates

SEED, FOLDS, VAL_RATIO = 42, 5, 0.1          # cross_validate.py 기본값 — 달라지면 2차 확률과 행이 안 맞는다
CANDIDATES = {"hashlr": "Hashing 2^18 char 1-3gram + LogReg", "tfidflr": "TF-IDF char_wb 2-4gram 20k + LogReg"}
MODEL_DIR = PROJECT_ROOT / "experiments/checkpoints"
LATENCY_OUT = GF.RESULTS / "linear_stage1_latency_srbh_4class.json"
OUTPUT = GF.RESULTS / "linear_stage1_srbh_4class.json"
FIGURE = PROJECT_ROOT / "docs/figures/cascade/stage1_alternatives_frontier_srbh_4class.png"
E1_CONDITION = "CPU b1, 1 thread, end-to-end, real length (E1)"
BALANCED = GF.RESULTS / "cvprobs_srbh_4class_cnn_raw_rgb_sealed_bal.npz"   # 언더샘플링 1차(T11), 없으면 건너뛴다
# E2 후보(1차 재설계). 채택 여부와 무관하게 같은 그래프에 올려 본다 — 사전 판정이 아니라 탐색이다.
E2_LABELS = {"cnn_crop8": "RGB CNN, top 8 rows", "cnn_gmp": "RGB CNN, global max pool", "cnn_1d": "RGB 1-D CNN"}
E2_LATENCY = GF.RESULTS / "latency_stage1_variants_srbh_4class.json"       # latency_fair.py --stage1-variants


def probs_path(name):
    return GF.RESULTS / f"cvprobs_srbh_4class_{name}_raw_sealed.npz"


def build_pipeline(name, n_classes):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    import baseline_tfidf

    vectorizer = (HashingVectorizer(analyzer="char", ngram_range=(1, 3), n_features=2 ** 18,
                                    alternate_sign=False, lowercase=False) if name == "hashlr" else
                  TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000, lowercase=False))
    return make_pipeline(vectorizer, baseline_tfidf.build_classifier("logreg", n_classes))


def fit(names):
    import joblib
    from sklearn.model_selection import StratifiedKFold

    texts, y, classes = CV.load_text_pool("srbh_4class", "raw", exclude_test=True)
    texts = np.asarray(texts, dtype=object)
    with np.load(GF.STAGE2, allow_pickle=False) as saved:
        second = dict(saved)
    if str(second["label_fingerprint"]) != CV.label_fingerprint(y):
        raise ValueError("텍스트 풀의 라벨 지문이 2차 CV 확률과 다릅니다.")
    folds = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    for name in names:
        rows = []
        for k, (trval, test) in enumerate(folds, start=1):
            train_rel, val_rel = CV.inner_train_val_split(y[trval], VAL_RATIO, SEED + k)
            train, val = trval[train_rel], trval[val_rel]
            # 2차와 같은 행을 봐야 넘김 비교가 성립한다 — 인덱스 하나라도 다르면 멈춘다.
            if not (np.array_equal(val, second[f"val_idx_{k}"]) and np.array_equal(test, second[f"test_idx_{k}"])):
                raise ValueError(f"fold {k} 분할이 2차 CV 확률과 다릅니다.")
            started = time.perf_counter()
            pipeline = build_pipeline(name, len(classes)).fit(texts[train], y[train])
            rows.append({"val_idx": val, "val_probs": pipeline.predict_proba(texts[val]),
                         "test_idx": test, "test_probs": pipeline.predict_proba(texts[test])})
            f1 = gates._macro_f1_fast(y[test], rows[-1]["test_probs"].argmax(1), len(classes))
            print(f"{name} fold {k}: test Macro-F1 {f1:.4f} ({time.perf_counter() - started:.0f}s)", flush=True)
            if k == 1:   # 지연 측정용. 가중치 값은 지연에 영향이 없어 한 fold 면 된다.
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, MODEL_DIR / f"linear_stage1_{name}.joblib")
        CV.save_fold_probabilities(probs_path(name), classes, y, rows)


def latency(names, repeats=30, n=2000, warmup=50):
    import joblib
    from threadpoolctl import threadpool_limits
    from src.eval import latency_fair as LF
    import data_text

    texts_all, _ = data_text.load_text_split(LF.TRACK, "val", LF.TEXT)
    picked = np.random.default_rng(LF.SEED).choice(len(texts_all), n, replace=False)   # E1 과 같은 표본
    texts = [texts_all[i] for i in picked]
    result = {"n": n, "repeats": repeats, "threads": 1, "method": "pipeline.predict_proba([text]) per request"}
    with threadpool_limits(limits=1):
        for name in names:
            pipeline = joblib.load(MODEL_DIR / f"linear_stage1_{name}.joblib")
            for text in texts[:warmup]:
                pipeline.predict_proba([text])
            times = np.empty((repeats, n), dtype=np.int64)
            for r in range(repeats):
                for i, text in enumerate(texts):
                    started = time.perf_counter_ns()
                    pipeline.predict_proba([text])
                    times[r, i] = time.perf_counter_ns() - started
            result[name] = LF.stats(times.ravel() / 1e6)
            print(f"{name}: {result[name]}", flush=True)
    LATENCY_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def q_statistic(first, second, folds):
    """Yule Q — 두 단계의 정오답이 얼마나 같이 움직이는가. 1 이면 같은 문제를 틀리고, 낮을수록 서로 보완한다."""
    right1, right2 = (np.concatenate([m[f"test_probs_{k}"].argmax(1) == first["y"][first[f"test_idx_{k}"]]
                                      for k in folds]) for m in (first, second))
    both, neither = (right1 & right2).sum(), (~right1 & ~right2).sum()
    only1, only2 = (right1 & ~right2).sum(), (~right1 & right2).sum()
    return float((both * neither - only1 * only2) / (both * neither + only1 * only2))


def load_stage1(path, strata):
    """1차 cvprobs 하나를 2차와 정렬 검증해 읽고, 지연과 무관한 것(게이트 경로·단독 F1·Q)을 한 번만 계산한다."""
    first, second, folds = validate_alignment(path, GF.STAGE2)
    paths, solo = GF.fold_paths(first, second, strata)
    n_classes = len(first["classes"])
    val_solo2 = [gates._macro_f1_fast(first["y"][first[f"val_idx_{k}"]], second[f"val_probs_{k}"].argmax(1), n_classes)
                 for k in folds]
    return {"paths": paths, "solo": solo, "val_solo2": val_solo2, "q": q_statistic(first, second, folds)}


def versus(base, other):
    """같은 운영점 규칙(표준 넘김 B0)의 π=3.3% 배수를 두 1차에서 fold 대응으로 비교한다. 못 고른 fold 가 있으면 None."""
    if "reason" in base or "reason" in other:
        return None
    s_base, s_other = (np.array(row["per_fold_speedup"]["b0"]) for row in (base, other))
    return {"speedup_base": float(s_base.mean()), "speedup_other": float(s_other.mean()),
            "speedup_gain": float(s_other.mean() / s_base.mean() - 1),
            "corrected_p": GC.corrected_t(s_other - s_base)["p"],
            "f1_diff_pp": float((other["b0"]["macro_f1"] - base["b0"]["macro_f1"]) * 100)}


def analyze():
    conditions = GC.latency_conditions()
    if E1_CONDITION not in conditions:
        raise ValueError("E1 지연 결과가 없습니다 — latency_fair.py 를 먼저 실행하세요.")
    ms1_cnn, ms2 = conditions[E1_CONDITION]
    linear = json.loads(LATENCY_OUT.read_text(encoding="utf-8"))
    stage1 = {"rgb_cnn": (GF.STAGE1, ms1_cnn, "RGB CNN (ours)"),
              **{name: (probs_path(name), linear[name]["p50"], label) for name, label in CANDIDATES.items()}}
    if E2_LATENCY.is_file():
        variant_ms = json.loads(E2_LATENCY.read_text(encoding="utf-8"))["stage1_e2e_ms"]
        # 변형의 지연은 "같은 실행에서 잰 현행 CNN 대비 비율 × E1 의 현행 지연"으로 맞춘다.
        # 실행마다 절대값이 4~6% 표류하므로(E1 1.31ms vs 변형 실행 1.26ms) 그대로 쓰면
        # 변형만 빨라 보인다 — 후보 간 차이는 번갈아 잰 같은 실행 안에서만 믿을 수 있다.
        same_run_cnn = variant_ms["cnn"]["p50"]
        for name, label in E2_LABELS.items():
            probs = GF.RESULTS / f"cvprobs_srbh_4class_{name}_raw_rgb_sealed.npz"
            if probs.is_file() and name in variant_ms:
                stage1[name] = (probs, ms1_cnn * variant_ms[name]["p50"] / same_run_cnn, label)
    strata = np.asarray(CV.load_pool_strata("srbh_4class", exclude_test=True))
    loaded = {name: load_stage1(path, strata) for name, (path, _, _) in stage1.items()}
    result = {"prereg": "docs/paper_revision/03_prereg_v2.md", "pi": GF.PI, "stage2_ms": ms2,
              "latency_condition": E1_CONDITION, "stage2_solo_macro_f1": float(loaded["rgb_cnn"]["solo"][1]),
              "stage1": {}}
    for name, (_, ms1, label) in stage1.items():
        L = loaded[name]
        result["stage1"][name] = {
            "label": label, "ms1": ms1, "upper_bound": ms2 / ms1, "solo_macro_f1": float(L["solo"][0]),
            "q_with_stage2": L["q"],
            **{view: GC.secondary_rule(L["paths"], L["val_solo2"], col, ms1, ms2) for view, col in GC.VIEWS}}
    # E3 판정: 같은 운영점 규칙에서 π=3.3% 배수가 RGB CNN 1차보다 20% 이상 높은가.
    #          E2 후보는 같은 비교를 하되 판정 깃발을 달지 않는다(탐색).
    for name in stage1:
        if name == "rgb_cnn":
            continue
        for view, _ in GC.VIEWS:
            row = versus(result["stage1"]["rgb_cnn"][view], result["stage1"][name][view])
            if row and name in CANDIDATES:
                row["heterogeneity_not_necessary"] = bool(row["speedup_gain"] >= 0.20)
            if row:
                result["stage1"][name][f"vs_rgb_cnn_{view}"] = row
    # T11: 1차 학습 방식 기준(class weight 가 언더샘플링보다 20% 이상 빠르고 F1 손실 0.3%p 이하)을 fold-test 로 다시 본다.
    #      지연은 두 1차가 같으므로(같은 구조) 차이는 넘김에서만 온다. 기준이 정해진 채움 기준 지연과 E1 지연 둘 다 낸다.
    if BALANCED.is_file():
        bal = load_stage1(BALANCED, strata)
        result["t11_stage1_training"] = {"solo_macro_f1_undersampled": float(bal["solo"][0]), "by_latency": {}}
        for condition in (GC.LEGACY_CPU[0], E1_CONDITION):
            t1, t2 = conditions[condition]
            cell = {}
            for view, col in GC.VIEWS:
                row = versus(GC.secondary_rule(bal["paths"], bal["val_solo2"], col, t1, t2),
                             GC.secondary_rule(loaded["rgb_cnn"]["paths"], loaded["rgb_cnn"]["val_solo2"], col, t1, t2))
                if row:
                    cell[view] = {**row, "criterion_met": bool(row["speedup_gain"] >= 0.20 and row["f1_diff_pp"] >= -0.3)}
            result["t11_stage1_training"]["by_latency"][condition] = cell
    curves = {name: (loaded[name]["paths"], ms1, label, loaded[name]["solo"]) for name, (_, ms1, label) in stage1.items()}
    plot(curves, ms2, result["stage2_solo_macro_f1"])
    result["figure"] = FIGURE.relative_to(PROJECT_ROOT).as_posix()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in result["stage1"].items():
        print(f"{name}: 1차 {row['ms1']:.4f}ms, 단독 F1 {row['solo_macro_f1']:.4f}, Q {row['q_with_stage2']:.3f}, "
              + ", ".join(f"{view} B0 {row[view]['b0']['speedup']:.3f}배/F1 {row[view]['b0']['macro_f1']:.4f}"
                          for view, _ in GC.VIEWS if "reason" not in row[view]))
        for view, _ in GC.VIEWS:
            if f"vs_rgb_cnn_{view}" in row:
                print(f"   판정({view}): {row[f'vs_rgb_cnn_{view}']}")
    for condition, cell in result.get("t11_stage1_training", {}).get("by_latency", {}).items():
        print(f"T11 [{condition}]: {cell}")


def plot(curves, ms2, solo2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    # 가장 느린 1차(해싱)의 상한 t2/t1 이 0.2 아래라 격자를 0.1 부터 잡는다 — 안 그러면 그 곡선이 통째로 빠진다.
    grid = np.linspace(0.1, max(ms2 / ms1 for _, ms1, _, _ in curves.values()), 300)
    for ax, (col, (_, title)) in zip(axes, enumerate(GF.NORMAL_VIEWS)):
        for (name, (paths, ms1, label, solo)), color in zip(curves.items(), ("tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown", "tab:cyan")):
            env = GF.envelopes({"b0": paths["b0"]}, ms1, ms2, col, grid)["b0"]
            reachable = ~np.isnan(env).any(0)
            ax.plot(grid[reachable], env[:, reachable].mean(0), color=color, lw=2,
                    label=f"stage 1 = {label} ({ms1:.3f} ms)")
            ax.fill_between(grid[reachable], env[:, reachable].min(0), env[:, reachable].max(0), color=color, alpha=.15)
            ax.scatter([ms2 / ms1], [solo[0]], marker="s", s=60, color=color, zorder=5)
        ax.scatter([1.0], [solo2], marker="*", s=180, c="k", zorder=6, label=f"char-CNN alone (F1 {solo2:.3f})")
        ax.axvline(1.0, color="k", lw=.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Speedup vs char-CNN alone (attack share 3.3%)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Test Macro-F1 (5-fold CV, band = fold min-max)")
    # 단독 점(네모)이 왼쪽 아래에 몰려 있어 범례를 축 안에 두면 가린다 — 그림 아래로 뺀다.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Cascade frontier by stage-1 model, standard confidence deferral (srbh_4class; squares = stage 1 alone)\n"
                 "CPU, batch 1, 1 thread, end-to-end latency", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    step = parser.add_mutually_exclusive_group(required=True)
    for name in ("fit", "latency", "analyze"):
        step.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--models", default=",".join(CANDIDATES), help="쉼표로 구분한 1차 후보")
    args = parser.parse_args()
    names = [name for name in args.models.split(",") if name]
    if set(names) - set(CANDIDATES):
        parser.error(f"--models 는 {sorted(CANDIDATES)} 중에서 고릅니다.")
    try:
        fit(names) if args.fit else latency(names) if args.latency else analyze()
    except (OSError, ValueError, KeyError) as error:
        print(f"[오류] E3 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
