"""E1 — 전처리를 포함한 공정 지연 기준선 (docs/paper_revision/03_prereg_v2.md §E1, §E1-G)

왜 필요한가:
    기존 지연(`cascade_tradeoff.py`)은 순전파만 쟀고, 2차 char-CNN 은 모든 요청을 2,304바이트로
    패딩해 쟀다. 요청 길이 중앙값은 96바이트라 2차 비용이 부풀려져 있을 수 있다. 실제 길이로
    돌린 2차가 1차(RGB 변환 + 순전파)보다 빠르면 "CPU·배치 1 속도 이득" 주장을 철회해야 한다.
    이 스크립트는 **판정을 내리지 않고** 판정에 쓰는 수치만 만든다 — 기준은 사전 문서에 고정돼 있다.

측정 단위:
    CPU 는 요청 1건씩(배치 1) 구성 요소별로 따로 잰다. 요청마다 길이가 달라 평균 하나로는
    꼬리가 안 보이므로 p50·p95·p99·평균을 모두 남긴다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "src" / "imaging"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from src.models import cascade, gate_signals  # noqa: E402  (cascade 가 src/models 를 경로에 넣는다)
import data_text  # noqa: E402
from payload_to_image import payload_to_rgb_image  # noqa: E402

TRACK, TEXT, SIDE, MAX_LEN = "srbh_4class", "raw", 48, 2304
# 2차의 가장 큰 합성곱 커널이 7이라, 그보다 짧은 입력은 커널 하나가 통째로 패딩만 보게 된다.
MIN_REAL_LEN = 7
PI = 0.033
SEED = 42
GPU_BATCHES = (32, 128, 1024)
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
# τ·r_A·r_N·클래스 순서의 출처. 권장 구성(1차 class weight + 2차 균형)의 val 선택 운영점이다.
TRADEOFF_JSON = RESULTS_DIR / ("tradeoff_srbh_4class_cnn_raw_rgb_s2bal_len2304_nall_s42"
                               "_dcuda-cpu_b1-32-128-1024.json")
OUT_JSON = RESULTS_DIR / "latency_fair_srbh_4class.json"
VARIANTS_JSON = RESULTS_DIR / "latency_stage1_variants_srbh_4class.json"

COMPONENTS = ("prep_img", "fwd1", "enc_seq_pad", "enc_seq_real", "fwd2_pad", "fwd2_real", "cascade")
COMPOSITES = {"stage1_e2e": ("prep_img", "fwd1"),
              "stage2_real_e2e": ("enc_seq_real", "fwd2_real"),
              "stage2_pad_e2e": ("enc_seq_pad", "fwd2_pad")}


def prep_img(text: str) -> torch.Tensor:
    """문자열 → (1,3,48,48) float 텐서. 학습 때의 `data_image.make_torch_dataset` 과 같은 값이어야 한다."""
    image = payload_to_rgb_image(text, side=SIDE)
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()


def enc_seq(text: str, pad: bool) -> torch.Tensor:
    """문자열 → 바이트+1 정수 텐서. pad=True 는 기존 방식(2,304 고정), False 는 실제 길이."""
    raw = text.encode("utf-8", errors="replace")[:MAX_LEN]
    seq = np.zeros(MAX_LEN if pad else max(len(raw), MIN_REAL_LEN), dtype=np.int64)
    if raw:
        seq[: len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
    return torch.from_numpy(seq).unsqueeze(0)


def time_request(net1, net2, text: str, tau: float):
    """요청 한 건의 구성 요소별 시간(ns)과 넘김 여부. 순서는 COMPONENTS 와 같다."""
    clock = time.perf_counter_ns
    t0 = clock()
    x_img = prep_img(text)
    t1 = clock()
    net1(x_img)
    t2 = clock()
    x_pad = enc_seq(text, True)
    t3 = clock()
    x_real = enc_seq(text, False)
    t4 = clock()
    net2(x_pad)
    t5 = clock()
    net2(x_real)
    t6 = clock()
    # 실제 배포 흐름을 한 번에 잰다 — 구성 요소 합과 다르면 그 차이가 곧 분기·softmax 비용이다.
    deferred = torch.softmax(net1(prep_img(text)), dim=1).max().item() < tau
    if deferred:
        net2(enc_seq(text, False))
    t7 = clock()
    return (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6), deferred


def stats(ms: np.ndarray) -> dict:
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(ms.mean())}


def speedup(t1: float, t2: float, escalation: float) -> float:
    return float(t2 / (t1 + escalation * t2))


def judge(summary: dict, sample_rate: float, e_pi: float) -> dict:
    """사전 문서 E1 의 판정 재료. 임계 비교만 하고 해석은 문서에서 한다."""
    p50 = {name: row["p50"] for name, row in summary.items()}
    formula = {}
    for basis in ("p50", "mean"):
        value = {name: row[basis] for name, row in summary.items()}
        for method, (first, second) in (("fair_e2e_real", ("stage1_e2e", "stage2_real_e2e")),
                                        ("legacy_fwd_pad", ("fwd1", "fwd2_pad"))):
            formula[f"{method}_{basis}"] = {
                "t1_ms": value[first], "t2_ms": value[second],
                "sample_rate": speedup(value[first], value[second], sample_rate),
                "pi_0.033": speedup(value[first], value[second], e_pi)}
    return {
        "e1a_withdraw": bool(p50["fwd2_real"] <= p50["stage1_e2e"]),
        "upper_bound_real": p50["stage2_real_e2e"] / p50["stage1_e2e"],
        "upper_bound_pad_fwd_only": p50["fwd2_pad"] / p50["fwd1"],
        "speedup_formula": formula,
        "speedup_measured": summary["stage2_real_e2e"]["mean"] / summary["cascade"]["mean"],
        "sample_deferral_rate": sample_rate,
    }


@torch.inference_mode()
def measure_cpu(net1, net2, texts: list[str], tau: float, threads: int, repeats: int,
                e_pi: float, warmup: int = 50) -> dict:
    torch.set_num_threads(threads)
    for text in texts[:warmup]:
        time_request(net1, net2, text, tau)
    times = np.empty((repeats, len(texts), len(COMPONENTS)), dtype=np.int64)
    deferred = np.zeros(len(texts), dtype=bool)
    for r in range(repeats):
        for i, text in enumerate(texts):
            times[r, i], deferred[i] = time_request(net1, net2, text, tau)
    ms = {name: times[:, :, k].ravel() / 1e6 for k, name in enumerate(COMPONENTS)}
    for name, parts in COMPOSITES.items():
        ms[name] = sum(ms[part] for part in parts)   # 요청 단위 합(같은 측정끼리 더한다)
    summary = {name: stats(values) for name, values in ms.items()}
    return {"threads": threads, "components": summary,
            "judgment": judge(summary, float(deferred.mean()), e_pi)}


@torch.inference_mode()
def equivalence(net2, texts: list[str], y: np.ndarray, n_classes: int) -> dict:
    """패딩이 예측을 바꾸는지 확인한다 — 바뀐다면 실제 길이 지연은 '다른 모델'의 지연이 된다."""
    device = torch.device("cpu")
    x_pad = torch.from_numpy(data_text.encode_byte_matrix(texts, MAX_LEN))
    pred_pad = cascade.predict_probs(net2, x_pad, device, 512).argmax(axis=1)
    pred_real = np.array([int(net2(enc_seq(text, False)).argmax()) for text in texts])
    agreement = float((pred_pad == pred_real).mean())
    return {"n": len(texts), "agreement": agreement, "equivalent": agreement >= 0.999,
            "macro_f1_pad": gate_signals._macro_f1_fast(y, pred_pad, n_classes),
            "macro_f1_real": gate_signals._macro_f1_fast(y, pred_real, n_classes)}


def measure_gpu(net1, net2, texts: list[str]) -> dict:
    """E1-G: 배치 추론 지연(요청당 ms, 30라운드 중앙값). 판정 없음."""
    if not torch.cuda.is_available():
        return {"skipped": "CUDA 를 쓸 수 없어 건너뜀"}
    device = torch.device("cuda")
    net1.to(device), net2.to(device)
    rng = np.random.default_rng(SEED)
    rows = []
    try:
        for batch in GPU_BATCHES:
            picked = [texts[i] for i in rng.choice(len(texts), batch, replace=batch > len(texts))]
            x_img = torch.cat([prep_img(text) for text in picked])
            longest = max(MIN_REAL_LEN, max(len(text.encode("utf-8", errors="replace")) for text in picked))
            inputs = {"stage1_ms": (net1, x_img),
                      "stage2_pad2304_ms": (net2, torch.from_numpy(data_text.encode_byte_matrix(picked, MAX_LEN))),
                      "stage2_batchmax_ms": (net2, torch.from_numpy(
                          data_text.encode_byte_matrix(picked, min(longest, MAX_LEN))))}
            row = {"batch": batch, "batch_max_len": min(longest, MAX_LEN)}
            for name, (net, x) in inputs.items():
                row[name] = cascade.measure_latency(net, x, device, batch, rounds=30, repeats=5)
            rows.append(row)
    finally:
        net1.cpu(), net2.cpu()
    return {"device_name": torch.cuda.get_device_name(device), "rows": rows}


@torch.inference_mode()
def measure_stage1_variants(texts: list[str], n_classes: int, repeats: int, warmup: int = 50) -> dict:
    """E2: 1차 후보의 요청당 지연(이미지 변환 + 순전파, CPU·배치 1·스레드 1).

    지연은 가중치 값과 무관하므로 무작위 초기화 모델로 잰다(CV 는 체크포인트를 남기지 않는다).
    현행 1차도 같은 실행에서 다시 재 후보와 같은 조건으로 비교한다.
    """
    from cnn import E2_VARIANTS, PayloadCNN, build_e2_variant

    torch.set_num_threads(1)
    clock = time.perf_counter_ns
    nets = {name: (PayloadCNN(n_classes, in_channels=3) if name == "cnn"
                   else build_e2_variant(name, n_classes)).eval() for name in ("cnn",) + tuple(E2_VARIANTS)}
    for net in nets.values():
        for text in texts[:warmup]:
            net(prep_img(text))
    ms = {name: np.empty((repeats, len(texts))) for name in nets}
    for r in range(repeats):
        # 반복마다 후보를 번갈아 잰다 — 후보별로 몰아서 재면 기기 상태(발열·백그라운드)의 표류가
        # 후보 간 차이로 읽힌다. 판정 조건이 "현행 이하"라 1% 차이도 결과를 가른다.
        for name, net in nets.items():
            for i, text in enumerate(texts):
                t0 = clock()
                net(prep_img(text))
                ms[name][r, i] = (clock() - t0) / 1e6
    out = {name: stats(values.ravel()) for name, values in ms.items()}
    for name, row in out.items():
        print(f"  {name:<10} p50 {row['p50']:.4f} ms", flush=True)
    return out


BREAKDOWN_JSON = RESULTS_DIR / "latency_charcnn_breakdown_srbh_4class.json"
BREAKDOWN_PARTS = ("enc", "emb", "conv3", "conv5", "conv7", "head", "aux3", "floor", "full")
# 관측한 가장 낮은 1차 넘김률(TF-IDF 1차, docs/14 §8.26). 사전 문서 E7 에 고정된 값이다.
E_REF = 0.053
GATE_SPEEDUP = 1.5


def charcnn_parts(net2, aux_head, x: torch.Tensor, clock=time.perf_counter_ns):
    """CharCNN.forward 를 단계별로 나눠 실행하며 시간(ns)을 잰다. 반환: (로짓, 단계별 ns).

    원 모델과 같은 연산 순서여야 분해 합이 실제 순전파를 대표한다(테스트가 출력 일치를 고정).
    aux3 는 DL-C(내부 조기 종료)의 보조 헤드 비용 — 커널 3 출력만 보고 종료 여부를 정한다.
    """
    t0 = clock()
    emb = net2.embedding(x).transpose(1, 2)
    t1 = clock()
    pooled, marks = [], []
    for conv in net2.convs:
        pooled.append(torch.relu(conv(emb)).max(dim=2).values)
        marks.append(clock())
    logits = net2.classifier(net2.dropout(torch.cat(pooled, dim=1)))
    t5 = clock()
    torch.softmax(aux_head(pooled[0]), dim=1).max().item()
    t6 = clock()
    t2, t3, t4 = marks
    return logits, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)


def breakdown_gates(p50: dict) -> dict:
    """사전 문서 E7 의 판정 재료. t2 = 인코딩 + 2차 전체 순전파."""
    t2 = p50["enc"] + p50["full"]
    t1_b = p50["enc"] + p50["floor"]
    t_front = p50["enc"] + p50["emb"] + p50["conv3"] + p50["aux3"]
    t_rest = p50["conv5"] + p50["conv7"] + p50["head"]
    s_b = t2 / (t1_b + E_REF * t2)
    s_c = t2 / (t_front + E_REF * t_rest)
    return {"t2_ms": t2, "e_ref": E_REF, "threshold": GATE_SPEEDUP,
            "dl_b": {"t1_floor_ms": t1_b, "speedup_ub": s_b, "pass": bool(s_b >= GATE_SPEEDUP)},
            "dl_c": {"t_front_ms": t_front, "t_rest_ms": t_rest, "speedup_ub": s_c,
                     "pass": bool(s_c >= GATE_SPEEDUP)},
            "parts_sum_minus_full_ms": sum(p50[k] for k in ("emb", "conv3", "conv5", "conv7", "head")) - p50["full"]}


@torch.inference_mode()
def measure_charcnn_breakdown(texts: list[str], n_classes: int, repeats: int, warmup: int = 50) -> dict:
    """E7: char-CNN 단계별 지연(CPU·배치 1·스레드 1) + 바닥 모델. 판정은 breakdown_gates."""
    torch.set_num_threads(1)
    clock = time.perf_counter_ns
    net2 = cascade.load_net("charcnn", n_classes, TRACK, TEXT, True, torch.device("cpu"))
    torch.manual_seed(SEED)
    aux_head = torch.nn.Linear(net2.convs[0].out_channels, n_classes).eval()
    # 가장 싼 딥러닝 1차의 하한: 임베딩 → 평균 → 선형. 이보다 싼 신경망 1차는 사실상 없다.
    floor_net = torch.nn.Sequential(torch.nn.Embedding(257, 16), _MeanOverTime(), torch.nn.Linear(16, n_classes)).eval()

    def one(text: str) -> np.ndarray:
        t0 = clock()
        x = enc_seq(text, False)
        t1 = clock()
        _, parts = charcnn_parts(net2, aux_head, x, clock)
        t2 = clock()
        floor_net(x)
        t3 = clock()
        net2(x)
        t4 = clock()
        return np.array((t1 - t0, *parts, t3 - t2, t4 - t3), dtype=np.int64)

    for text in texts[:warmup]:
        one(text)
    times = np.empty((repeats, len(texts), len(BREAKDOWN_PARTS)), dtype=np.int64)
    for r in range(repeats):
        for i, text in enumerate(texts):
            times[r, i] = one(text)
    summary = {name: stats(times[:, :, k].ravel() / 1e6) for k, name in enumerate(BREAKDOWN_PARTS)}
    return {"threads": 1, "components": summary,
            "gates": breakdown_gates({name: row["p50"] for name, row in summary.items()})}


class _MeanOverTime(torch.nn.Module):
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return emb.mean(dim=1)


def environment(args) -> dict:
    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_hash = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        ram_gb = None
    return {"date": datetime.now().isoformat(timespec="seconds"), "git_hash": git_hash,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ram_gb": ram_gb, "seed": SEED, "args": vars(args)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000, help="val 에서 뽑는 요청 수")
    parser.add_argument("--repeats", type=int, default=30, help="표본 전체 반복 횟수")
    parser.add_argument("--threads", default="1,4,8", help="쉼표로 구분한 CPU 스레드 수")
    parser.add_argument("--tau", type=float, help="넘김 임계값. 생략하면 tradeoff JSON 의 운영점")
    parser.add_argument("--gpu", action="store_true", help="E1-G: GPU 배치 지연도 잰다")
    parser.add_argument("--stage1-variants", action="store_true",
                        help="E2: 1차 후보(cnn·cnn_crop8·cnn_gmp·cnn_1d)의 변환+순전파 지연만 잰다")
    parser.add_argument("--charcnn-breakdown", action="store_true",
                        help="E7: char-CNN 단계별 지연과 DL-B·DL-C 관문(스레드 1)만 잰다")
    parser.add_argument("--smoke", action="store_true", help="n=50·2회·1스레드, 파일 저장 없음")
    args = parser.parse_args()
    if args.smoke:
        args.n, args.repeats, args.threads = 50, 2, "1"
    try:
        args.threads = [int(value) for value in args.threads.split(",")]
    except ValueError:
        parser.error("--threads 는 쉼표로 구분한 양의 정수여야 합니다.")
    if min(args.threads) <= 0 or args.n <= 0 or args.repeats <= 0:
        parser.error("--n, --repeats, --threads 는 양수여야 합니다.")
    return args


def run(args) -> dict:
    if not TRADEOFF_JSON.exists():
        raise FileNotFoundError(
            f"운영점 JSON 이 없습니다: {TRADEOFF_JSON.relative_to(PROJECT_ROOT)}\n"
            "  → python src/eval/cascade_tradeoff.py --balance --stage1-no-balance 로 먼저 만드세요.")
    tradeoff = json.loads(TRADEOFF_JSON.read_text(encoding="utf-8"))
    classes = tradeoff["classes"]
    tau = args.tau if args.tau is not None else tradeoff["operating_point"]["tau"]
    r_a, r_n = tradeoff["reweighted"]["r_A"], tradeoff["reweighted"]["r_N"]
    e_pi = PI * r_a + (1 - PI) * r_n

    texts_all, labels = data_text.load_text_split(TRACK, "val", TEXT)
    if args.n > len(texts_all):
        raise ValueError(f"--n({args.n:,}) 이 val 크기({len(texts_all):,})보다 큽니다.")
    y_all = data_text.encode_labels_with(labels, classes)
    picked = np.random.default_rng(SEED).choice(len(texts_all), args.n, replace=False)
    texts = [texts_all[i] for i in picked]

    if args.stage1_variants:
        return {"env": environment(args), "n": args.n, "repeats": args.repeats,
                "stage1_e2e_ms": measure_stage1_variants(texts, len(classes), args.repeats)}
    if args.charcnn_breakdown:
        return {"env": environment(args), "n": args.n, "repeats": args.repeats,
                "breakdown": measure_charcnn_breakdown(texts, len(classes), args.repeats)}

    device = torch.device("cpu")
    net1 = cascade.load_net("cnn", len(classes), TRACK, TEXT, False, device, channels="rgb", in_channels=3)
    net2 = cascade.load_net("charcnn", len(classes), TRACK, TEXT, True, device)

    default_threads = torch.get_num_threads()
    lengths = [len(text.encode("utf-8", errors="replace")) for text in texts]
    result = {"env": environment(args), "tau": tau, "r_A": r_a, "r_N": r_n, "e_pi": e_pi,
              "n": args.n, "repeats": args.repeats,
              "sample_byte_length": stats(np.asarray(lengths, dtype=float)), "cpu": []}
    for threads in args.threads:
        print(f"CPU·배치 1 측정: 스레드 {threads}", flush=True)
        result["cpu"].append(measure_cpu(net1, net2, texts, tau, threads, args.repeats, e_pi))

    torch.set_num_threads(default_threads)
    print("패딩 대 실제 길이 동등성 확인", flush=True)
    if args.smoke:
        result["equivalence"] = equivalence(net2, texts, y_all[picked], len(classes))
    else:
        result["equivalence"] = equivalence(net2, texts_all, y_all, len(classes))
    if args.gpu:
        print("GPU 배치 지연 측정", flush=True)
        result["gpu"] = measure_gpu(net1, net2, texts_all)
    return result


def report(result: dict) -> None:
    for row in result["cpu"]:
        print(f"\n[스레드 {row['threads']}] 요청당 ms (p50 / p95 / p99 / 평균)")
        for name, s in row["components"].items():
            print(f"  {name:<16} {s['p50']:.4f} / {s['p95']:.4f} / {s['p99']:.4f} / {s['mean']:.4f}")
        j = row["judgment"]
        fair = j["speedup_formula"]["fair_e2e_real_p50"]
        legacy = j["speedup_formula"]["legacy_fwd_pad_p50"]
        print(f"  e1a_withdraw={j['e1a_withdraw']}  upper_bound_real={j['upper_bound_real']:.3f}"
              f"  upper_bound_pad_fwd_only={j['upper_bound_pad_fwd_only']:.3f}")
        print(f"  speedup_formula(p50, pi=3.3%): 공정 {fair['pi_0.033']:.3f}배 · 기존 방식 {legacy['pi_0.033']:.3f}배")
        print(f"  speedup_measured={j['speedup_measured']:.3f}배 (표본 넘김률 {j['sample_deferral_rate']:.1%})")
    eq = result["equivalence"]
    print(f"\n동등성: 일치율 {eq['agreement']:.4%} (n={eq['n']:,}, equivalent={eq['equivalent']})"
          f"  Macro-F1 패딩 {eq['macro_f1_pad']:.4f} · 실제 길이 {eq['macro_f1_real']:.4f}")
    gpu = result.get("gpu")
    if gpu:
        print("\nGPU 요청당 ms:", gpu.get("skipped") or gpu["device_name"])
        for row in gpu.get("rows", []):
            print(f"  배치 {row['batch']:>4}: 1차 {row['stage1_ms']:.5f} · 2차 2304패딩 {row['stage2_pad2304_ms']:.5f}"
                  f" · 2차 배치 최대 길이({row['batch_max_len']}) {row['stage2_batchmax_ms']:.5f}")


def report_breakdown(row: dict) -> None:
    print("\n[스레드 1] char-CNN 단계별 요청당 ms (p50 / p95 / 평균)")
    for name, s in row["components"].items():
        print(f"  {name:<6} {s['p50']:.4f} / {s['p95']:.4f} / {s['mean']:.4f}")
    g = row["gates"]
    print(f"  t2={g['t2_ms']:.4f} ms · 분해 합 - 전체 = {g['parts_sum_minus_full_ms']:+.4f} ms · E_ref={g['e_ref']:.1%}")
    print(f"  DL-B: 하한 t1 {g['dl_b']['t1_floor_ms']:.4f} ms → 상한 {g['dl_b']['speedup_ub']:.3f}배 (통과={g['dl_b']['pass']})")
    print(f"  DL-C: 앞 {g['dl_c']['t_front_ms']:.4f} · 나머지 {g['dl_c']['t_rest_ms']:.4f} ms"
          f" → 상한 {g['dl_c']['speedup_ub']:.3f}배 (통과={g['dl_c']['pass']})")


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"[오류] {error}", file=sys.stderr)
        sys.exit(1)
    if args.stage1_variants:
        if not args.smoke:
            VARIANTS_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"저장: {VARIANTS_JSON.relative_to(PROJECT_ROOT)}")
        return
    if args.charcnn_breakdown:
        report_breakdown(result["breakdown"])
        if not args.smoke:
            BREAKDOWN_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"저장: {BREAKDOWN_JSON.relative_to(PROJECT_ROOT)}")
        return
    report(result)
    if args.smoke:
        print("\n스모크 실행이라 결과 파일은 저장하지 않습니다.")
        return
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
