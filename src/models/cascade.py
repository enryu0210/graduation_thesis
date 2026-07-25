"""
Phase 9 — 하이브리드 캐스케이드 탐지기 (제안 CNN 1차 필터 + char-CNN 2차 판정)

왜 '융합'이 아니라 '캐스케이드'인가 — 설계의 핵심 근거:
    RQ1 실측(docs/04 §5)은 두 사실을 동시에 보여줬다.
      · 제안 CNN : clean Macro-F1 최하위(payload_4class 0.9496 / csicnorm 0.967)지만
                   학습 4.0s/epoch 로 char-CNN(29.6s) 대비 7배 이상 저렴하다.
      · char-CNN : 최고 정확도(0.995)지만 길이 2304 시퀀스를 3개 커널로 훑어 비용이 크다.
    이 둘을 흔한 방식대로 **특징 융합(two-branch: 이미지 CNN + char-CNN → concat → FC)** 하면
    모든 입력이 두 브랜치를 **다 통과**하므로 비용이 C_cnn + C_charcnn 이 된다.
    → 정확도는 오르지만 "제안 CNN 의 속도 장점"은 사라진다(오히려 char-CNN 단독보다 느림).

    그래서 여기서는 **선택적 실행(cascade)** 을 택한다:
        1차: 모든 트래픽을 값싼 제안 CNN 이 판정한다.
        2차: 1차 확신도(max softmax)가 임계값 τ 미만인 **소수 샘플만** char-CNN 이 재판정한다.
    평균 비용 = C_cnn + (에스컬레이션 비율 r) × C_charcnn 이므로, r 이 작으면
    "정확도는 char-CNN 급, 지연은 CNN 급"이 성립한다. τ 를 움직이면 정확도-지연 곡선(Pareto)이
    그려지고, τ→0 은 CNN 단독, τ→1 은 char-CNN 단독으로 수렴한다(두 단독 모델이 곡선의 양 끝점).
    문헌 근거: 저비용 1차 필터 + 2D-CNN 2차 판정의 하이브리드 구성(docs/thetics 의
    "시그니처 기반 필터링과 2D-CNN을 활용한 하이브리드 악성 트래픽 탐지 기법").

방법론적 주의 — τ 는 반드시 val 에서 고른다:
    test 로 τ 를 고르면 임계값이 test 에 과적합되어 성능이 부풀려진다(정보 누수).
    이 스크립트는 val 스윕으로 τ 를 확정한 뒤 **그 τ 를 test 에 한 번만** 적용한다.

산출물:
    experiments/results/cascade_{track}_{text}[_bal].json     — 선택된 τ·지표·지연·스윕 곡선
    experiments/results/pred_{track}_cascade_{text}[_bal].npz — 샘플 단위 예측(상보성 분석용)
    docs/figures/models/cascade_{track}_{text}[_bal].png      — 정확도/지연 vs 에스컬레이션 곡선
    docs/figures/models/cm_{track}_cascade_{text}[_bal].png   — 혼동행렬

사용법 (GPU 권장 — 두 모델 체크포인트가 먼저 있어야 함):
    python src/models/train.py --model cnn     --track payload_4class_csicnorm --balance
    python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance
    python src/models/cascade.py --track payload_4class_csicnorm --balance
    python src/models/cascade.py --track payload_4class_csicnorm --balance --smoke   # 동작 점검
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 저장 가능
import matplotlib.pyplot as plt

# 이웃 모듈 import 경로 설정(train.py 와 동일 규칙)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import torch  # noqa: E402

import metrics as M  # noqa: E402
import data_image  # noqa: E402
import data_text  # noqa: E402
from train import DEFAULT_LR  # noqa: E402  (체크포인트 tag 의 lr 규칙을 train.py 와 공유)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "models"
CKPT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"

# 2차(정밀) 판정기 후보. 1차는 제안 CNN 으로 고정한다(제안 모델이 주인공이어야 하므로).
STAGE2_MODELS = ("charcnn", "bilstm")

# BiLSTM 은 긴 시퀀스(2304)를 순환 처리해 활성값 메모리가 배치×길이로 폭증한다.
# run_evasion.py 에서 배치 1024 로 CUDA OOM(30GiB+)이 났던 전례가 있어 여기서도 작게 잡는다.
_BATCH_BY_MODEL = {"bilstm": 128}
_DEFAULT_BATCH = 512


# ---------------------------------------------------------------------------
# 체크포인트 로드 / 확률 추론
# ---------------------------------------------------------------------------
def checkpoint_tag(track: str, model: str, text: str, balance: bool,
                   channels: str = "gray", encoders=None, lr: float = DEFAULT_LR) -> str:
    """train.py 의 저장 태그 규칙을 재현한다(파일명 불일치 방지).

    train.py 규칙: {track}_{model}_{text}{채널}{패치}{lr}[_bal]
    - 채널 접미사는 `data_image._channel_suffix` 를 **그대로 재사용**한다. 직접 '_rgb' 를
      만들면 RGB ablation 조합(`_rgb-rb-cc-bd` 등)을 못 찾는다(단일 진실 소스 유지).
    - lr 은 기본값이면 생략(train.py 와 동일) — ViT 용 비기본 lr 체크포인트도 가리킬 수 있게 둔다.
    - 패치 접미사는 vit 전용이라 여기서는 해당 없음(1차는 cnn, 2차는 텍스트 모델).
    """
    ch_tag = data_image._channel_suffix(channels, encoders) if model == "cnn" else ""
    lr_tag = "" if lr == DEFAULT_LR else f"_lr{lr:g}"
    return f"{track}_{model}_{text}{ch_tag}{lr_tag}" + ("_bal" if balance else "")


def load_net(model: str, num_classes: int, track: str, text: str, balance: bool,
             device, channels: str = "gray", in_channels: int = 1,
             encoders=None, lr: float = DEFAULT_LR):
    """학습된 체크포인트를 로드해 eval 모드 모델을 반환한다.

    체크포인트가 없으면 '어떤 명령으로 만들면 되는지'까지 알려주는 에러를 낸다
    (이 스크립트는 학습을 하지 않고 기존 산출물을 재사용하는 것이 설계 의도다).
    """
    if model == "cnn":
        import cnn as cnn_mod
        net = cnn_mod.build_model(num_classes, in_channels=in_channels)
    else:
        import text_models
        net = text_models.build_model(model, num_classes)

    path = CKPT_DIR / f"{checkpoint_tag(track, model, text, balance, channels, encoders, lr)}.pt"
    if not path.exists():
        rgb_hint = ""
        if model == "cnn" and channels == "rgb":
            rgb_hint = f" --channels rgb --rgb-encoders {','.join(encoders or ())}"
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {path}\n"
            f"  → 먼저 학습하세요: python src/models/train.py --model {model} "
            f"--track {track}{' --balance' if balance else ''}{rgb_hint}"
            f"{'' if lr == DEFAULT_LR else f' --lr {lr:g}'}"
        )
    net.load_state_dict(torch.load(path, map_location=device))
    return net.to(device).eval()


@torch.no_grad()
def predict_probs(net, x: torch.Tensor, device, batch: int) -> np.ndarray:
    """(N, ...) 입력 텐서에 대해 softmax 확률 (N, K) 를 배치 추론한다."""
    out = []
    for i in range(0, len(x), batch):
        logits = net(x[i:i + batch].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def build_inputs(track: str, split: str, text: str, side: int, channels: str,
                 encoders, max_len: int, limit: int | None):
    """한 split 에 대해 (이미지 텐서, 시퀀스 텐서, 라벨, 클래스명) 을 만든다.

    두 표현이 **같은 샘플·같은 순서**여야 캐스케이드가 성립하므로, 이미지 npz 의 라벨과
    CSV 의 라벨이 일치하는지 방어적으로 검증한다(트랙 재생성 시점이 어긋나면 조용히
    잘못된 결과가 나오는 것을 막는다).
    """
    imgs, y_img, classes = data_image.load_split(track, split, text, side, channels, encoders)
    texts, labels_str = data_text.load_text_split(track, split, text)
    y_txt = data_text.encode_labels_with(labels_str, classes)

    if len(imgs) != len(texts):
        raise ValueError(
            f"[{split}] 이미지({len(imgs):,})와 CSV({len(texts):,}) 샘플 수가 다릅니다. "
            f"preprocess.py 와 build_image_dataset.py 를 같은 시점 데이터로 다시 실행하세요."
        )
    if not np.array_equal(y_img, y_txt):
        raise ValueError(
            f"[{split}] 이미지 라벨과 CSV 라벨이 어긋납니다(행 순서/클래스 매핑 불일치). "
            f"data/processed 와 data/images 를 함께 재생성하세요."
        )

    if limit:
        imgs, texts, y_img = imgs[:limit], texts[:limit], y_img[:limit]

    # 이미지: uint8 → float(0~1) + 채널축 정리를 data_image 규칙으로 일원화
    x_img = data_image.make_torch_dataset(imgs, y_img).tensors[0]
    x_seq = torch.from_numpy(data_text.encode_byte_matrix(texts, max_len))
    return x_img, x_seq, y_img, classes


# ---------------------------------------------------------------------------
# 캐스케이드 규칙 + τ 스윕
# ---------------------------------------------------------------------------
def cascade_apply(p1: np.ndarray, p2: np.ndarray, tau: float):
    """1차 확률 p1, 2차 확률 p2 를 임계값 τ 로 합친다.

    규칙: 1차 확신도(max softmax) < τ 인 샘플만 2차 결과로 대체한다.
    τ=0 → 아무도 넘기지 않음(제안 CNN 단독), τ>1 → 전부 넘김(char-CNN 단독).

    반환: (예측, 결합 확률, 에스컬레이션 마스크)
    ※ 여기서는 평가를 위해 p2 를 전 샘플에 대해 미리 계산해 두지만, 실제 배포에서는
      에스컬레이션된 샘플에만 2차 모델을 돌린다(비용 계산은 그 가정으로 한다).
    """
    escalate = p1.max(axis=1) < tau
    probs = np.where(escalate[:, None], p2, p1)
    return probs.argmax(axis=1), probs, escalate


def sweep_taus(p1: np.ndarray, p2: np.ndarray, y_true: np.ndarray,
               classes: list[str], taus: np.ndarray) -> list[dict]:
    """τ 격자를 훑어 (에스컬레이션 비율, 정확도 지표) 곡선을 만든다.

    스윕은 τ 개수만큼 반복되므로 PR-AUC/ROC-AUC 같은 무거운 점수 지표는 계산하지 않는다
    (최종 운영점에서만 전체 지표를 낸다).
    """
    rows = []
    for tau in taus:
        pred, _, escalate = cascade_apply(p1, p2, tau)
        m = M.compute_metrics(y_true, pred, classes)
        af = m.get("attack_focused") or {}
        rows.append({
            "tau": float(tau),
            "escalation_rate": float(escalate.mean()),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "mcc": m["mcc"],
            "benign_evasion_rate": af.get("benign_evasion_rate"),
            "normal_false_positive_rate": af.get("normal_false_positive_rate"),
        })
    return rows


def make_tau_grid(conf: np.ndarray, n: int = 41) -> np.ndarray:
    """확신도 분위수로 τ 격자를 만든다.

    왜 분위수인가: 확신도는 대부분 1.0 근처에 몰려 있어 균등 격자(linspace)로는
    에스컬레이션 비율이 0%→100% 로 급변해 곡선이 안 그려진다. 분위수를 쓰면
    에스컬레이션 비율이 고르게 퍼진 곡선을 얻는다.
    """
    qs = np.quantile(conf, np.linspace(0.0, 1.0, n))
    # 0.0(=CNN 단독)과 1.01(=char-CNN 단독) 양 끝점을 반드시 포함시킨다.
    return np.unique(np.concatenate([[0.0], qs, [1.01]]))


def select_tau(val_rows: list[dict], rule: str, budget: float,
               teacher_macro_f1: float) -> dict:
    """val 스윕 결과에서 운영 τ 를 고른다(test 는 절대 보지 않는다).

    - rule='match-teacher' : val Macro-F1 이 2차 모델(char-CNN) 단독 이상이 되는 τ 중
      **에스컬레이션이 가장 작은** 것. 즉 "정확도는 최소한 char-CNN 만큼, 비용은 최소로".
      끝내 못 따라잡으면 val Macro-F1 최대 지점으로 대체한다(동률이면 더 싼 쪽).
    - rule='budget'        : 에스컬레이션 ≤ budget 제약 아래 val Macro-F1 최대.
      (SLA 로 지연 예산이 먼저 정해진 배포 상황을 상정한 규칙)
    """
    if rule == "budget":
        feasible = [r for r in val_rows if r["escalation_rate"] <= budget]
        if not feasible:  # budget 이 0 에 가까우면 가장 싼 지점으로
            feasible = [min(val_rows, key=lambda r: r["escalation_rate"])]
        best = max(feasible, key=lambda r: (r["macro_f1"], -r["escalation_rate"]))
        return {"rule": rule, "budget": budget, **best}

    reached = [r for r in val_rows if r["macro_f1"] >= teacher_macro_f1]
    if reached:
        best = min(reached, key=lambda r: (r["escalation_rate"], -r["macro_f1"]))
        note = "val Macro-F1 이 2차 모델 단독 이상이 되는 최소 에스컬레이션 지점"
    else:
        best = max(val_rows, key=lambda r: (r["macro_f1"], -r["escalation_rate"]))
        note = "2차 모델 단독을 못 따라잡아 val Macro-F1 최대 지점으로 대체"
    return {"rule": rule, "teacher_macro_f1": teacher_macro_f1, "note": note, **best}


def oracle_point(p1: np.ndarray, p2: np.ndarray, y_true: np.ndarray,
                 classes: list[str]) -> dict:
    """상한(oracle) 라우팅: '1차가 틀린 샘플만' 정확히 2차로 넘겼다면?

    어떤 확신도 게이트도 이보다 잘할 수 없으므로, 실제 게이트가 상한에 얼마나 근접했는지를
    재는 기준선이 된다(게이트 품질 = 확신도가 오답을 얼마나 잘 골라내는가).
    """
    escalate = p1.argmax(axis=1) != y_true
    pred = np.where(escalate, p2.argmax(axis=1), p1.argmax(axis=1))
    m = M.compute_metrics(y_true, pred, classes)
    return {
        "escalation_rate": float(escalate.mean()),
        "macro_f1": m["macro_f1"],
        "mcc": m["mcc"],
    }


# ---------------------------------------------------------------------------
# 지연 측정
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_latency(net, x: torch.Tensor, device, batch: int,
                    warmup: int = 5, repeats: int = 20, rounds: int = 5) -> float:
    """샘플당 순전파 지연(ms)을 측정한다. **여러 라운드의 중앙값**을 반환한다.

    - warmup: 첫 호출에는 CUDA 커널 로딩·메모리 할당 비용이 섞여 과대측정되므로 버린다.
    - GPU 는 비동기 실행이라 synchronize() 없이 재면 시간이 0 에 가깝게 나온다.
    - ⚠️ 왜 중앙값인가: 단발 측정은 GPU 클럭 부스트 상태·다른 프로세스 간섭에 따라
      실행마다 크게 흔들린다(실측: char-CNN 이 같은 조건에서 0.0555~0.0904 ms, 약 60% 변동).
      이 Phase 의 헤드라인이 "같은 정확도를 몇 분의 1 비용으로"라 지연이 곧 결론이므로,
      라운드별 측정치의 중앙값을 써서 이상치 하나가 speedup 을 부풀리지 못하게 막는다.
    - 전처리(텍스트→이미지/바이트) 비용은 제외한 **모델 순전파 기준**이다(두 모델 모두
      동일 기준이라 비교는 공정하며, 전처리는 µs 수준으로 결론을 바꾸지 않는다).
    """
    sample = x[:batch].to(device)
    for _ in range(warmup):
        net(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    per_round = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(repeats):
            net(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_round.append(elapsed / (repeats * len(sample)) * 1000.0)
    return float(np.median(per_round))


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def fig_pareto(test_rows: list[dict], op: dict, oracle: dict,
               ms1: float, ms2: float, out_path: Path, title: str) -> None:
    """에스컬레이션 비율 vs (정확도 / 지연) 2패널 그림. 라벨은 ASCII 만(한글 폰트 없음)."""
    x = [r["escalation_rate"] for r in test_rows]
    f1 = [r["macro_f1"] for r in test_rows]
    order = np.argsort(x)
    x = np.asarray(x)[order]
    f1 = np.asarray(f1)[order]

    # 곡선의 양 끝점이 곧 두 단독 모델의 성능이다(τ=0 → CNN, τ>1 → char-CNN).
    f1_stage1, f1_stage2 = f1[0], f1[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(x, f1, "o-", color="#2980b9", lw=2, ms=4, label="cascade")
    ax.axhline(f1_stage1, ls="--", color="#7f8c8d", lw=1,
               label=f"stage-1 only (CNN) = {f1_stage1:.4f}")
    ax.axhline(f1_stage2, ls=":", color="#c0392b", lw=1.5,
               label=f"stage-2 only (char-CNN) = {f1_stage2:.4f}")
    ax.plot([oracle["escalation_rate"]], [oracle["macro_f1"]], "*", color="#f39c12",
            ms=14, label=f"oracle routing = {oracle['macro_f1']:.4f}")
    ax.plot([op["escalation_rate"]], [op["macro_f1"]], "D", color="#27ae60", ms=9,
            label=f"operating point (tau={op['tau']:.4f})")
    ax.set_xlabel("escalation rate (fraction sent to stage-2)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("accuracy vs escalation", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    lat = ms1 + x * ms2
    ax.plot(x, lat, "o-", color="#8e44ad", lw=2, ms=4, label="cascade avg latency")
    ax.axhline(ms1, ls="--", color="#7f8c8d", lw=1, label=f"stage-1 only = {ms1:.4f} ms")
    ax.axhline(ms1 + ms2, ls="-.", color="#16a085", lw=1,
               label=f"two-branch fusion (both always) = {ms1 + ms2:.4f} ms")
    ax.axhline(ms2, ls=":", color="#c0392b", lw=1.5, label=f"stage-2 only = {ms2:.4f} ms")
    op_lat = ms1 + op["escalation_rate"] * ms2
    ax.plot([op["escalation_rate"]], [op_lat], "D", color="#27ae60", ms=9,
            label=f"operating point = {op_lat:.4f} ms")
    ax.set_xlabel("escalation rate (fraction sent to stage-2)")
    ax.set_ylabel("avg inference latency (ms/sample)")
    ax.set_title("latency vs escalation", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(
        description="하이브리드 캐스케이드(제안 CNN 1차 + char-CNN 2차) 평가")
    p.add_argument("--track", default="payload_4class_csicnorm",
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary",
                            "ustc_flow_binary"])
    p.add_argument("--stage2", default="charcnn", choices=STAGE2_MODELS,
                   help="2차 정밀 판정기(1차는 제안 CNN 고정)")
    p.add_argument("--text", default="raw", choices=["raw", "decoded"])
    p.add_argument("--balance", action="store_true",
                   help="'_bal' 체크포인트(균형 학습본)를 로드. csicnorm 트랙 기본 권장")
    p.add_argument("--side", type=int, default=48)
    p.add_argument("--channels", default="gray", choices=["gray", "rgb"],
                   help="1차 CNN 의 입력 채널(빌드·학습 때와 동일해야 함). "
                        "docs/04 §5 기준 정확도 최고 CNN 은 rgb rb/cc/bd 조합")
    p.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy",
                   help="rgb 일 때 R,G,B 인코더(학습 때와 동일해야 체크포인트를 찾음)")
    p.add_argument("--lr", type=float, default=DEFAULT_LR,
                   help="체크포인트 탐색용 학습률(기본값이면 tag 에 안 붙음 — train.py 규칙)")
    p.add_argument("--max-len", type=int, default=48 * 48)
    p.add_argument("--select", default="match-teacher", choices=["match-teacher", "budget"],
                   help="τ 선택 규칙(val 기준). budget 은 --budget 과 함께 사용")
    p.add_argument("--budget", type=float, default=0.10,
                   help="--select budget 일 때 허용 에스컬레이션 비율 상한")
    p.add_argument("--n-taus", type=int, default=41, help="τ 격자 개수(분위수 기반)")
    p.add_argument("--limit", type=int, default=None, help="샘플 수 제한(스모크용)")
    p.add_argument("--smoke", action="store_true", help="빠른 동작 확인(작게, 저장 생략)")
    args = p.parse_args()

    if args.smoke:
        args.limit = args.limit or 500
        args.n_taus = min(args.n_taus, 11)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    print(f"=== 캐스케이드 평가: track={args.track} stage2={args.stage2} device={device} ===")

    # 1) 입력 준비(val: τ 선택용 / test: 최종 보고용)
    x_img_va, x_seq_va, y_va, classes = build_inputs(
        args.track, "val", args.text, args.side, args.channels, encoders,
        args.max_len, args.limit)
    x_img_te, x_seq_te, y_te, _ = build_inputs(
        args.track, "test", args.text, args.side, args.channels, encoders,
        args.max_len, args.limit)
    in_channels = x_img_te.shape[1]
    print(f"  val={len(y_va):,} test={len(y_te):,} classes={classes} in_channels={in_channels}")

    # 2) 두 모델 로드
    net1 = load_net("cnn", len(classes), args.track, args.text, args.balance,
                    device, args.channels, in_channels, encoders, args.lr)
    net2 = load_net(args.stage2, len(classes), args.track, args.text, args.balance, device)
    b1 = _DEFAULT_BATCH
    b2 = _BATCH_BY_MODEL.get(args.stage2, _DEFAULT_BATCH)

    # 3) 두 모델의 확률을 미리 계산(평가 편의용. 배포 시 2차는 에스컬레이션분만 실행)
    p1_va = predict_probs(net1, x_img_va, device, b1)
    p2_va = predict_probs(net2, x_seq_va, device, b2)
    p1_te = predict_probs(net1, x_img_te, device, b1)
    p2_te = predict_probs(net2, x_seq_te, device, b2)

    # 4) val 스윕 → τ 확정 (test 는 아직 보지 않는다)
    taus = make_tau_grid(p1_va.max(axis=1), args.n_taus)
    val_rows = sweep_taus(p1_va, p2_va, y_va, classes, taus)
    teacher_val_f1 = val_rows[-1]["macro_f1"]  # τ 최대 = 전부 2차 = char-CNN 단독
    op_val = select_tau(val_rows, args.select, args.budget, teacher_val_f1)
    tau = op_val["tau"]
    print(f"  [τ 선택/val] rule={args.select} → tau={tau:.6f} "
          f"escalation={op_val['escalation_rate']:.4f} macroF1={op_val['macro_f1']:.4f} "
          f"(char-CNN 단독 val macroF1={teacher_val_f1:.4f})")

    # 5) test 에 τ 를 한 번만 적용 → 최종 지표
    pred_te, prob_te, esc_te = cascade_apply(p1_te, p2_te, tau)
    result = M.compute_metrics(y_te, pred_te, classes, y_score=prob_te)

    # 6) 지연 측정 + 비용 모델
    ms1 = measure_latency(net1, x_img_te, device, min(b1, len(x_img_te)))
    ms2 = measure_latency(net2, x_seq_te, device, min(b2, len(x_seq_te)))
    esc_rate = float(esc_te.mean())
    ms_cascade = ms1 + esc_rate * ms2
    result["latency"] = {
        "stage1_ms_per_sample": ms1,
        "stage2_ms_per_sample": ms2,
        "cascade_ms_per_sample": ms_cascade,
        "fusion_ms_per_sample": ms1 + ms2,   # 두 브랜치를 항상 다 태우는 융합형(비교군)
        "speedup_vs_stage2_only": ms2 / ms_cascade if ms_cascade > 0 else None,
        "overhead_vs_stage1_only": ms_cascade / ms1 if ms1 > 0 else None,
        "note": "모델 순전파 기준(전처리 제외). 배포 가정: 2차는 에스컬레이션분만 실행.",
    }

    # 7) 비교 기준선: 두 단독 모델 + oracle 라우팅
    test_rows = sweep_taus(p1_te, p2_te, y_te, classes, taus)
    m1 = M.compute_metrics(y_te, p1_te.argmax(axis=1), classes)
    m2 = M.compute_metrics(y_te, p2_te.argmax(axis=1), classes)
    oracle = oracle_point(p1_te, p2_te, y_te, classes)

    result["model"] = "cascade"
    result["escalation_rate"] = esc_rate
    result["selected_tau"] = tau
    result["tau_selection"] = op_val
    result["oracle_routing"] = oracle
    result["stage_alone"] = {
        "stage1_cnn": {"macro_f1": m1["macro_f1"], "mcc": m1["mcc"],
                       "accuracy": m1["accuracy"],
                       "attack_focused": m1.get("attack_focused")},
        f"stage2_{args.stage2}": {"macro_f1": m2["macro_f1"], "mcc": m2["mcc"],
                                  "accuracy": m2["accuracy"],
                                  "attack_focused": m2.get("attack_focused")},
    }
    result["sweep_val"] = val_rows
    result["sweep_test"] = test_rows
    result["config"] = {
        "track": args.track, "stage1": "cnn", "stage2": args.stage2, "text": args.text,
        "side": args.side, "channels": args.channels, "max_len": args.max_len,
        "balance": args.balance, "device": str(device), "limit": args.limit,
        "select": args.select, "budget": args.budget, "n_taus": int(len(taus)),
    }

    # 8) 콘솔 요약
    print("\n  [test 결과 — 단독 vs 캐스케이드]")
    print(f"    stage-1 CNN 단독     : macroF1={m1['macro_f1']:.4f} MCC={m1['mcc']:.4f} "
          f"| {ms1:.4f} ms/sample")
    print(f"    stage-2 {args.stage2:<12}: macroF1={m2['macro_f1']:.4f} MCC={m2['mcc']:.4f} "
          f"| {ms2:.4f} ms/sample")
    print(f"    캐스케이드           : macroF1={result['macro_f1']:.4f} "
          f"MCC={result['mcc']:.4f} | {ms_cascade:.4f} ms/sample "
          f"(에스컬레이션 {esc_rate:.2%})")
    print(f"    oracle 라우팅(상한)  : macroF1={oracle['macro_f1']:.4f} "
          f"(에스컬레이션 {oracle['escalation_rate']:.2%})")
    print(f"    → 2차 단독 대비 {result['latency']['speedup_vs_stage2_only']:.2f}배 빠름, "
          f"1차 단독 대비 {result['latency']['overhead_vs_stage1_only']:.2f}배 비용")
    print("  " + M.format_summary("cascade", result))

    if args.smoke:
        print("\n  [smoke] 저장 생략(코드 동작 확인만).")
        return

    # 9) 저장 — train.py 명명 관습 계승(모델명 자리에 'cascade')
    # ⚠️ 실험을 가르는 설정을 tag 에 전부 반영한다. 안 그러면 서로 덮어쓴다
    #    (커밋 5eede2f 사고: RGB 조합이 전부 '_rgb' 로 저장돼 상호 덮어쓰기).
    #    캐스케이드에서 실험을 가르는 축: 2차 모델 · 1차 채널 · lr · τ 선택 규칙 · 균형화.
    bal = "_bal" if args.balance else ""
    s2 = "" if args.stage2 == "charcnn" else f"-{args.stage2}"  # 기본 조합은 접미사 없이
    ch_tag = data_image._channel_suffix(args.channels, encoders)
    lr_tag = "" if args.lr == DEFAULT_LR else f"_lr{args.lr:g}"
    sel_tag = "" if args.select == "match-teacher" else f"_b{args.budget:g}"
    tag = f"{args.track}_cascade{s2}_{args.text}{ch_tag}{lr_tag}{sel_tag}{bal}"
    M.save_report(result, RESULTS_DIR / f"{tag}.json")
    M.save_predictions(y_te, pred_te, classes, RESULTS_DIR / f"pred_{tag}.npz", y_score=prob_te)
    M.save_confusion_matrix(y_te, pred_te, classes, FIG_DIR / f"cm_{tag}.png",
                            title=f"cascade cnn->{args.stage2} ({args.track})")
    fig_pareto(test_rows, {"tau": tau, "escalation_rate": esc_rate,
                           "macro_f1": result["macro_f1"]}, oracle, ms1, ms2,
               FIG_DIR / f"cascade_curve_{tag}.png",
               f"Hybrid cascade: CNN (stage-1) -> {args.stage2} (stage-2), {args.track}")
    print(f"\n  [저장] 지표 → experiments/results/{tag}.json")
    print(f"  [저장] 예측 → experiments/results/pred_{tag}.npz")
    print(f"  [저장] 그림 → docs/figures/models/cascade_curve_{tag}.png, cm_{tag}.png")


if __name__ == "__main__":
    main()
