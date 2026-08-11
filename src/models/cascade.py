"""
Phase 9 — 하이브리드 캐스케이드 탐지기 (RGB CNN 1차 필터 + char-CNN 2차 판정)

왜 '융합'이 아니라 '캐스케이드'인가 — 설계의 핵심 근거:
    RQ1 실측(docs/04 §5)은 두 사실을 동시에 보여줬다.
      · RGB CNN : clean Macro-F1 최하위(payload_4class 0.9496 / csicnorm 0.967)지만
                   학습 4.0s/epoch 로 char-CNN(29.6s) 대비 7배 이상 저렴하다.
      · char-CNN : 최고 정확도(0.995)지만 길이 2304 시퀀스를 3개 커널로 훑어 비용이 크다.
    이 둘을 흔한 방식대로 **특징 융합(two-branch: 이미지 CNN + char-CNN → concat → FC)** 하면
    모든 입력이 두 브랜치를 **다 통과**하므로 비용이 C_cnn + C_charcnn 이 된다.
    → 정확도는 오르지만 "RGB CNN 의 속도 장점"은 사라진다(오히려 char-CNN 단독보다 느림).

    그래서 여기서는 **선택적 실행(cascade)** 을 택한다:
        1차: 모든 트래픽을 값싼 RGB CNN 이 판정한다.
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
from tagging import (  # noqa: E402  (tag 규칙 단일 진실 소스 — 사본을 만들지 않는다)
    DEFAULT_AUG_RATIO, DEFAULT_AUX_WEIGHT, DEFENSE_MODES, build_tag,
    aux_weight_suffix, defense_suffix, exit_suffix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "models"
CKPT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"

# 2차(정밀) 판정기 후보. 1차는 이미지 CNN 계열로 고정한다(캐스케이드의 값싼 필터 역할).
STAGE2_MODELS = ("charcnn", "bilstm")
# 1차 후보: 일반 RGB CNN 과 조기종료 CNN(Phase 12/M4). 둘 다 같은 이미지 입력을 받는다.
STAGE1_MODELS = ("cnn", "cnn_ee")
# H7-1 의 정확도 조건(docs/11 §7): 단일 split 실행 간 노이즈 ±0.11pp 를 그대로 쓴다.
EXIT_MCC_TOLERANCE = 0.0011

# BiLSTM 은 긴 시퀀스(2304)를 순환 처리해 활성값 메모리가 배치×길이로 폭증한다.
# run_evasion.py 에서 배치 1024 로 CUDA OOM(30GiB+)이 났던 전례가 있어 여기서도 작게 잡는다.
_BATCH_BY_MODEL = {"bilstm": 128}
_DEFAULT_BATCH = 512


# ---------------------------------------------------------------------------
# 체크포인트 로드 / 확률 추론
# ---------------------------------------------------------------------------
def checkpoint_tag(track: str, model: str, text: str, balance: bool,
                   channels: str = "gray", encoders=None, lr: float = DEFAULT_LR,
                   defense: str = "none", mutation_split: str | None = None,
                   aug_ratio: float | None = None,
                   aux_weight: float | None = None) -> str:
    """train.py 의 저장 태그 규칙을 그대로 따른다(파일명 불일치 방지).

    ⚠️ 규칙을 여기서 다시 구현하지 않고 `tagging.build_tag` 에 위임한다. 예전에는 이 함수가
    규칙 사본을 들고 있었는데, 방어 축이 추가되면서 사본이 어긋나면 **서로 다른 실험이 같은
    파일을 덮어쓰는** 실패 모드가 재발한다(커밋 5eede2f, docs/09 §9.7). tagging.py 가 단일
    진실 소스다.

    채널 접미사는 1차(이미지 CNN 계열)에만 붙는다 — 2차는 텍스트 모델이라 채널 축이 없다.
    보조 헤드 가중치(_aw)도 마찬가지로 조기종료 1차에만 붙는다.
    """
    is_stage1_image = model in STAGE1_MODELS
    return build_tag(track, model, text,
                     channels=channels if is_stage1_image else "gray",
                     encoders=encoders if is_stage1_image else None,
                     lr=lr, balance=balance, defense=defense,
                     mutation_split=mutation_split, aug_ratio=aug_ratio,
                     aux_weight=aux_weight if model == "cnn_ee" else None)


def load_net(model: str, num_classes: int, track: str, text: str, balance: bool,
             device, channels: str = "gray", in_channels: int = 1,
             encoders=None, lr: float = DEFAULT_LR,
             defense: str = "none", mutation_split: str | None = None,
             aug_ratio: float | None = None, aux_weight: float | None = None):
    """학습된 체크포인트를 로드해 eval 모드 모델을 반환한다.

    체크포인트가 없으면 '어떤 명령으로 만들면 되는지'까지 알려주는 에러를 낸다
    (이 스크립트는 학습을 하지 않고 기존 산출물을 재사용하는 것이 설계 의도다).
    """
    if model == "cnn":
        import cnn as cnn_mod
        net = cnn_mod.build_model(num_classes, in_channels=in_channels)
    elif model == "cnn_ee":
        import cnn as cnn_mod
        # 임계값은 여기서 정하지 않는다 — val 스윕이 고른 값을 나중에 net.exit_threshold 로 넣는다.
        net = cnn_mod.build_early_exit_model(num_classes, in_channels=in_channels)
    else:
        import text_models
        net = text_models.build_model(model, num_classes)

    path = CKPT_DIR / f"{checkpoint_tag(track, model, text, balance, channels, encoders, lr, defense, mutation_split, aug_ratio, aux_weight)}.pt"
    if not path.exists():
        rgb_hint = ""
        if model in STAGE1_MODELS and channels == "rgb":
            rgb_hint = f" --channels rgb --rgb-encoders {','.join(encoders or ())}"
        def_hint = ""
        if defense != "none":
            def_hint = f" --defense {defense}"
            if defense == "advtrain":
                def_hint += f" --mutation-split {mutation_split} --aug-ratio {aug_ratio:g}"
        aw_hint = ""
        if model == "cnn_ee" and aux_weight is not None and aux_weight != DEFAULT_AUX_WEIGHT:
            aw_hint = f" --aux-weight {aux_weight:g}"
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {path}\n"
            f"  → 먼저 학습하세요: python src/models/train.py --model {model} "
            f"--track {track}{' --balance' if balance else ''}{rgb_hint}{def_hint}{aw_hint}"
            f"{'' if lr == DEFAULT_LR else f' --lr {lr:g}'}"
        )
    net.load_state_dict(torch.load(path, map_location=device))
    return net.to(device).eval()


@torch.no_grad()
def predict_probs(net, x: torch.Tensor, device, batch: int) -> np.ndarray:
    """(N, ...) 입력 텐서에 대해 softmax 확률 (N, K) 를 배치 추론한다."""
    return predict_probs_with_exits(net, x, device, batch)[0]


@torch.no_grad()
def predict_probs_with_exits(net, x: torch.Tensor, device, batch: int):
    """확률 (N,K) 와 **조기종료 단계** (N,) 를 함께 반환한다(조기종료 모델이 아니면 단계는 None).

    왜 따로 모으나: 모델은 마지막 forward 의 종료 단계만 속성에 남긴다. 배치 루프를 돌면
    마지막 배치 것만 남아 전체 분포를 잃는다 — 비용 회계가 조용히 틀리게 되므로 여기서 이어붙인다.
    """
    out, stages = [], []
    for i in range(0, len(x), batch):
        logits = net(x[i:i + batch].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
        stage = getattr(net, "last_exit_stage", None)
        if stage is not None:
            stages.append(stage.cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
    exits = np.concatenate(stages) if stages else None
    return probs, exits


# ---------------------------------------------------------------------------
# 조기종료(M4) — 임계값은 val 에서 고르고 test 에 1회 적용한다(τ 와 같은 원칙)
# ---------------------------------------------------------------------------
def make_exit_grid(net, x_img, device, batch: int, n: int = 15) -> np.ndarray:
    """첫 블록 헤드의 확신도 분위수로 조기종료 임계값 격자를 만든다.

    왜 분위수인가: τ 격자와 같은 이유다. 확신도가 1.0 근처에 몰려 있어 균등 격자로는
    종료 비율이 0%↔100% 로 급변해 곡선이 안 그려진다.
    임계값 0(전원 첫 블록 종료)과 NO_EARLY_EXIT(아무도 종료 안 함) 양 끝점을 반드시 넣는다 —
    이 두 끝점이 각각 '가장 싼 극단'과 '기존 3블록 CNN'이라 비교 기준이 된다.
    """
    import cnn as cnn_mod

    saved = net.exit_threshold
    net.exit_threshold = 0.0          # 전원이 1블록에서 확정 → 반환 확률이 곧 1블록 헤드 확률
    probs, _ = predict_probs_with_exits(net, x_img, device, batch)
    net.exit_threshold = saved

    qs = np.quantile(probs.max(axis=1), np.linspace(0.0, 1.0, n))
    return np.unique(np.concatenate([[0.0], qs, [cnn_mod.NO_EARLY_EXIT]]))


def sweep_exit_thresholds(net, x_img, y_true: np.ndarray, classes: list[str],
                          device, batch: int, grid: np.ndarray) -> list[dict]:
    """조기종료 임계값 격자를 훑어 (정확도, 종료 분포, 평균 깊이) 곡선을 만든다.

    평균 깊이(mean_exit_depth)는 비용의 대리 지표다. 실제 지연은 선택된 지점에서만 재고
    (측정이 비싸다), 선택 자체는 이 대리 지표로 한다 — 블록이 적게 돌수록 싸다는 관계는
    단조라 순위가 뒤바뀌지 않는다.
    """
    import cnn as cnn_mod

    saved = net.exit_threshold
    rows = []
    for thr in grid:
        net.exit_threshold = float(thr)
        probs, exits = predict_probs_with_exits(net, x_img, device, batch)
        m = M.compute_metrics(y_true, probs.argmax(axis=1), classes)
        rows.append({
            "exit_threshold": float(thr),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "mcc": m["mcc"],
            "exit_rates": cnn_mod.exit_distribution(exits, net.n_stages),
            "mean_exit_depth": float(np.mean(exits) + 1.0),  # 1-based 블록 수
        })
    net.exit_threshold = saved
    return rows


def select_exit_threshold(rows: list[dict], tolerance: float = EXIT_MCC_TOLERANCE) -> dict:
    """정확도를 지키는 선에서 가장 싼 임계값을 고른다(val 기준, test 는 보지 않는다).

    규칙: '조기종료를 끈 지점'의 MCC 대비 하락이 tolerance(기본 0.11pp = 단일 split 노이즈)
    이내인 지점 중 **평균 깊이가 최소**인 것. 동률이면 MCC 가 높은 쪽.

    ⚠️ 이 규칙은 H7-1(같은 test MCC 유지하에 1차 비용 ≥30% 감소)을 val 에서 미리 강제한다.
    조건을 만족하는 지점이 하나도 없으면 조기종료를 **끄는 것**이 정답이므로 그 지점을 돌려준다
    (docs/08 의 hybrid 처럼 "시도했으나 미채택"이 정직한 결과다).
    """
    baseline = max(rows, key=lambda r: r["exit_threshold"])  # 임계값 최대 = 조기종료 없음
    feasible = [r for r in rows if baseline["mcc"] - r["mcc"] <= tolerance]
    if not feasible:
        feasible = [baseline]
    best = min(feasible, key=lambda r: (r["mean_exit_depth"], -r["mcc"]))
    return {
        "rule": f"val MCC drop <= {tolerance:g} 중 평균 깊이 최소",
        "baseline_no_exit_mcc": baseline["mcc"],
        "selected": best,
        "adopted": best["exit_threshold"] < baseline["exit_threshold"],
    }


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
    τ=0 → 아무도 넘기지 않음(RGB CNN 단독), τ>1 → 전부 넘김(char-CNN 단독).

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
                    warmup: int = 5, repeats: int = 20, rounds: int = 5,
                    sample_idx: np.ndarray | None = None) -> float:
    """샘플당 순전파 지연(ms)을 측정한다. **여러 라운드의 중앙값**을 반환한다.

    - warmup: 첫 호출에는 CUDA 커널 로딩·메모리 할당 비용이 섞여 과대측정되므로 버린다.
    - GPU 는 비동기 실행이라 synchronize() 없이 재면 시간이 0 에 가깝게 나온다.
    - ⚠️ 왜 중앙값인가: 단발 측정은 GPU 클럭 부스트 상태·다른 프로세스 간섭에 따라
      실행마다 크게 흔들린다(실측: char-CNN 이 같은 조건에서 0.0555~0.0904 ms, 약 60% 변동).
      이 Phase 의 헤드라인이 "같은 정확도를 몇 분의 1 비용으로"라 지연이 곧 결론이므로,
      라운드별 측정치의 중앙값을 써서 이상치 하나가 speedup 을 부풀리지 못하게 막는다.
    - 전처리(텍스트→이미지/바이트) 비용은 제외한 **모델 순전파 기준**이다(두 모델 모두
      동일 기준이라 비교는 공정하며, 전처리는 µs 수준으로 결론을 바꾸지 않는다).
    - ⚠️ sample_idx: **조기종료 모델에서는 배치 구성이 곧 비용**이다(쉬운 샘플이 몰린 배치는
      빨리 끝난다). test 셋이 클래스순으로 정렬돼 있으면 앞 batch 개만 재는 기본 동작이
      한 클래스에 편향돼 비용을 왜곡한다. 그래서 조기종료 모델에는 시드 고정 무작위 인덱스를
      넘겨 대표성 있는 배치를 쓴다. 일반 모델은 비용이 입력과 무관하므로 기존 동작을 유지한다
      (과거 실측치와의 비교 가능성 보존).
    """
    sample = (x[:batch] if sample_idx is None else x[sample_idx]).to(device)
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
        description="하이브리드 캐스케이드(RGB CNN 1차 + char-CNN 2차) 평가")
    p.add_argument("--track", default="payload_4class_csicnorm",
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary",
                            "ustc_flow_binary"])
    p.add_argument("--stage1", default="cnn", choices=STAGE1_MODELS,
                   help="1차 필터. cnn=일반 RGB CNN, cnn_ee=다단 조기종료 CNN(Phase 12/M4). "
                        "cnn_ee 는 조기종료 임계값을 val 에서 골라 test 에 1회 적용한다")
    p.add_argument("--stage2", default="charcnn", choices=STAGE2_MODELS,
                   help="2차 정밀 판정기")
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
    # 방어 축(Phase 11). 캐스케이드가 제안 모델이 되면서, 방어 학습본으로 구성한 캐스케이드도
    # 평가 대상이 됐다(docs/10 §0.1). 1차·2차 **양쪽 모두** 같은 방어 축의 체크포인트를 쓴다 —
    # 시스템 전체를 방어한 구성이어야 "제안 시스템의 강건성"을 말할 수 있기 때문이다.
    p.add_argument("--defense", default="none", choices=list(DEFENSE_MODES),
                   help="1·2차 체크포인트를 어떤 방어 학습본으로 쓸지. τ 는 그 구성의 val 에서 다시 고른다")
    p.add_argument("--mutation-split", default=None, choices=["S0", "SA"],
                   help="advtrain 체크포인트의 변형 분할(학습 때 쓴 값과 일치해야 함)")
    p.add_argument("--aug-ratio", type=float, default=None,
                   help=f"advtrain 체크포인트의 증강 비율(기본 {DEFAULT_AUG_RATIO})")
    # ── Phase 12 (M4) 조기종료 축 ─────────────────────────────────────────
    p.add_argument("--aux-weight", type=float, default=DEFAULT_AUX_WEIGHT,
                   help="cnn_ee 체크포인트의 보조 헤드 가중치(학습 때와 같아야 찾는다)")
    p.add_argument("--exit-threshold", type=float, default=None,
                   help="조기종료 임계값을 직접 지정(기본은 val 스윕으로 선택). "
                        "직접 지정하면 결과 tag 에 '_ex' 가 붙어 별도 파일로 남는다")
    p.add_argument("--exit-tolerance", type=float, default=EXIT_MCC_TOLERANCE,
                   help="임계값 선택 시 허용할 val MCC 하락폭(기본 0.0011 = 단일 split 노이즈)")
    p.add_argument("--n-exits", type=int, default=15, help="조기종료 임계값 격자 개수")
    args = p.parse_args()

    # 방어 축 정합성 — 어긋난 채로 돌면 엉뚱한 체크포인트를 조용히 집어 결과가 오염된다.
    if args.defense == "advtrain":
        if args.mutation_split is None:
            p.error("--defense advtrain 은 --mutation-split {S0,SA} 가 필요합니다(실험을 가르는 축)")
        if args.aug_ratio is None:
            args.aug_ratio = DEFAULT_AUG_RATIO
    elif args.mutation_split is not None or args.aug_ratio is not None:
        p.error("--mutation-split/--aug-ratio 는 --defense advtrain 일 때만 의미가 있습니다")
    if args.defense == "norm" and args.text != "decoded":
        p.error("--defense norm 은 --text decoded 와 함께 써야 합니다(정규화본으로 학습된 체크포인트)")
    # 조기종료 축은 cnn_ee 에서만 의미가 있다 — 일반 CNN 에 붙이면 tag 가 실험을 잘못 표현한다.
    if args.stage1 != "cnn_ee" and args.exit_threshold is not None:
        p.error("--exit-threshold 는 --stage1 cnn_ee 일 때만 의미가 있습니다")

    if args.smoke:
        args.limit = args.limit or 500
        args.n_taus = min(args.n_taus, 11)
        args.n_exits = min(args.n_exits, 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    print(f"=== 캐스케이드 평가: track={args.track} stage1={args.stage1} "
          f"stage2={args.stage2} device={device} ===")

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
    net1 = load_net(args.stage1, len(classes), args.track, args.text, args.balance,
                    device, args.channels, in_channels, encoders, args.lr,
                    args.defense, args.mutation_split, args.aug_ratio,
                    aux_weight=args.aux_weight if args.stage1 == "cnn_ee" else None)
    net2 = load_net(args.stage2, len(classes), args.track, args.text, args.balance, device,
                    defense=args.defense, mutation_split=args.mutation_split,
                    aug_ratio=args.aug_ratio)
    b1 = _DEFAULT_BATCH
    b2 = _BATCH_BY_MODEL.get(args.stage2, _DEFAULT_BATCH)

    # 2-b) 조기종료 임계값 확정 — **val 에서 고르고 test 에 1회 적용**(τ 와 같은 원칙).
    #      순서가 중요하다: 1차의 종료 임계값이 확신도 분포를 바꾸고, 그 분포가 곧 게이트
    #      입력이라 τ 도 이 구성에서 다시 골라야 한다(방어 arm 에서와 같은 이유).
    exit_info = None
    if args.stage1 == "cnn_ee":
        if args.exit_threshold is not None:
            net1.exit_threshold = args.exit_threshold
            exit_info = {"rule": "cli", "selected": {"exit_threshold": args.exit_threshold},
                         "adopted": True}
            print(f"  [조기종료] 임계값 직접 지정 → {args.exit_threshold:g}")
        else:
            grid = make_exit_grid(net1, x_img_va, device, b1, args.n_exits)
            exit_rows = sweep_exit_thresholds(net1, x_img_va, y_va, classes, device, b1, grid)
            exit_info = select_exit_threshold(exit_rows, args.exit_tolerance)
            exit_info["sweep_val"] = exit_rows
            net1.exit_threshold = exit_info["selected"]["exit_threshold"]
            sel = exit_info["selected"]
            print(f"  [조기종료/val] {exit_info['rule']} → threshold={sel['exit_threshold']:.6f} "
                  f"MCC={sel['mcc']:.4f}(무종료 {exit_info['baseline_no_exit_mcc']:.4f}) "
                  f"평균깊이={sel['mean_exit_depth']:.3f} "
                  f"종료분포={[round(r, 4) for r in sel['exit_rates']]}")
            if not exit_info["adopted"]:
                print("  ⚠️ 정확도 조건을 만족하는 조기종료 지점이 없어 **조기종료를 끈** 지점이 선택됨")

    # 3) 두 모델의 확률을 미리 계산(평가 편의용. 배포 시 2차는 에스컬레이션분만 실행)
    p1_va = predict_probs(net1, x_img_va, device, b1)
    p2_va = predict_probs(net2, x_seq_va, device, b2)
    # 1차 test 는 종료 단계까지 함께 받는다(조기종료가 아니면 exits_te 는 None).
    p1_te, exits_te = predict_probs_with_exits(net1, x_img_te, device, b1)
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
    # ⚠️ 조기종료 1차는 **배치 구성이 곧 비용**이라 대표성 있는 무작위 배치로 잰다(측정 편향 방지).
    n_lat = min(b1, len(x_img_te))
    sample_idx = None
    if args.stage1 == "cnn_ee":
        sample_idx = np.random.default_rng(42).choice(len(x_img_te), size=n_lat, replace=False)
    ms1 = measure_latency(net1, x_img_te, device, n_lat, sample_idx=sample_idx)
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

    # 6-b) M4 판정 — 같은 가중치에서 조기종료만 껐다 켠 A/B(H7-1·H7-2, docs/11 §7)
    if args.stage1 == "cnn_ee":
        import cnn as cnn_mod

        adopted_thr = net1.exit_threshold
        # 조기종료를 켠 상태의 1차 단독 지표(= 아래 step 7 의 m1 과 같은 값. 재추론하지 않는다)
        m1_exit = M.compute_metrics(y_te, p1_te.argmax(axis=1), classes)

        # 무종료 기준선: 가중치는 그대로, 임계값만 꺼서 잰다 → 조기종료 순수 효과가 분리된다.
        net1.exit_threshold = cnn_mod.NO_EARLY_EXIT
        p1_te_noexit = predict_probs(net1, x_img_te, device, b1)
        m1_noexit = M.compute_metrics(y_te, p1_te_noexit.argmax(axis=1), classes)
        ms1_noexit = measure_latency(net1, x_img_te, device, n_lat, sample_idx=sample_idx)
        net1.exit_threshold = adopted_thr  # 원상복구(이후 코드가 운영 임계값을 전제한다)

        cost_cut = (ms1_noexit - ms1) / ms1_noexit if ms1_noexit > 0 else None
        mcc_delta = m1_exit["mcc"] - m1_noexit["mcc"]
        speedup = result["latency"]["speedup_vs_stage2_only"]
        ms_cascade_noexit = ms1_noexit + esc_rate * ms2
        result["early_exit"] = {
            "threshold": adopted_thr,
            "aux_weight": args.aux_weight,
            "selection": exit_info,
            "exit_rates_test": cnn_mod.exit_distribution(exits_te, net1.n_stages),
            "mean_exit_depth_test": float(np.mean(exits_te) + 1.0),
            "stage1_mcc_with_exit": m1_exit["mcc"],
            "stage1_mcc_no_exit": m1_noexit["mcc"],
            "stage1_ms_no_exit": ms1_noexit,
            "stage1_cost_reduction": cost_cut,
            # 같은 τ·같은 에스컬레이션에서 조기종료만 뺀 반사실 — 캐스케이드 속도 이득의 순수분
            "cascade_ms_no_exit": ms_cascade_noexit,
            "speedup_vs_stage2_only_no_exit": (ms2 / ms_cascade_noexit
                                               if ms_cascade_noexit > 0 else None),
            "verdict": {
                # ⚠️ docs/11 §7 의 문구는 "같은 test MCC(±0.11pp 이내) 유지"지만, 여기서는
                #    **하락이 0.11pp 이내**라는 단측 조건으로 읽는다. 조기종료가 정확도를
                #    올렸다는 이유로 기각하는 것은 명백한 사양 결함이기 때문이다. val 선택
                #    규칙도 처음부터 단측(하락만 제한)이었으므로 이쪽이 일관된다.
                #    이 해석은 **실측 전에** 확정했다(docs/11 §12.3).
                "H7-1": {
                    "criterion": f"MCC 하락 <= {args.exit_tolerance:g} and 1차 비용 감소 >= 30%",
                    "mcc_delta": mcc_delta, "cost_reduction": cost_cut,
                    "result": ("충족" if (mcc_delta >= -args.exit_tolerance
                                        and cost_cut is not None and cost_cut >= 0.30)
                               else "미충족"),
                },
                "H7-2": {
                    "criterion": "대표 운영점 speedup >= 6.5배",
                    "speedup": speedup,
                    "result": "충족" if (speedup is not None and speedup >= 6.5) else "미충족",
                },
            },
            "note": ("조기종료는 확정된 샘플을 배치에서 실제로 빼고 남은 것만 다음 블록에 넣는다"
                     "(마스킹이 아니라 축소 배치) — 그래서 측정 지연이 실제로 줄어든다."),
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
        f"stage1_{args.stage1}": {"macro_f1": m1["macro_f1"], "mcc": m1["mcc"],
                                  "accuracy": m1["accuracy"],
                                  "attack_focused": m1.get("attack_focused")},
        f"stage2_{args.stage2}": {"macro_f1": m2["macro_f1"], "mcc": m2["mcc"],
                                  "accuracy": m2["accuracy"],
                                  "attack_focused": m2.get("attack_focused")},
    }
    result["sweep_val"] = val_rows
    result["sweep_test"] = test_rows
    result["config"] = {
        "track": args.track, "stage1": args.stage1, "stage2": args.stage2, "text": args.text,
        "side": args.side, "channels": args.channels, "max_len": args.max_len,
        "balance": args.balance, "device": str(device), "limit": args.limit,
        "select": args.select, "budget": args.budget, "n_taus": int(len(taus)),
        # 방어 구성을 기록해 둔다. run_evasion 이 이 파일에서 τ 를 읽을 때, 어떤 구성의
        # 운영점인지 확인할 수 있어야 한다(엉뚱한 τ 를 재사용하는 사고 방지).
        "defense": {"mode": args.defense,
                    "mutation_split": args.mutation_split if args.defense == "advtrain" else None,
                    "aug_ratio": args.aug_ratio if args.defense == "advtrain" else None},
    }

    # 8) 콘솔 요약
    print("\n  [test 결과 — 단독 vs 캐스케이드]")
    print(f"    stage-1 {args.stage1:<12}: macroF1={m1['macro_f1']:.4f} MCC={m1['mcc']:.4f} "
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
    if "early_exit" in result:
        ee = result["early_exit"]
        print(f"\n  [M4 조기종료 — 같은 가중치에서 켰다/껐다 비교]")
        print(f"    1차 비용 : {ee['stage1_ms_no_exit']:.4f} → {ms1:.4f} ms/sample "
              f"({ee['stage1_cost_reduction']:.1%} 감소)")
        print(f"    1차 MCC  : {ee['stage1_mcc_no_exit']:.4f} → {ee['stage1_mcc_with_exit']:.4f} "
              f"(Δ{ee['verdict']['H7-1']['mcc_delta']:+.4f})")
        print(f"    종료분포 : {[round(r, 4) for r in ee['exit_rates_test']]} "
              f"(평균 깊이 {ee['mean_exit_depth_test']:.3f}블록)")
        print(f"    캐스케이드 speedup: {ee['speedup_vs_stage2_only_no_exit']:.2f}배(무종료) → "
              f"{result['latency']['speedup_vs_stage2_only']:.2f}배")
        print(f"    [판정] H7-1={ee['verdict']['H7-1']['result']} / "
              f"H7-2={ee['verdict']['H7-2']['result']}")
    print("  " + M.format_summary("cascade", result))

    if args.smoke:
        print("\n  [smoke] 저장 생략(코드 동작 확인만).")
        return

    # 9) 저장 — train.py 명명 관습 계승(모델명 자리에 'cascade')
    # ⚠️ 실험을 가르는 설정을 tag 에 전부 반영한다. 안 그러면 서로 덮어쓴다
    #    (커밋 5eede2f 사고: RGB 조합이 전부 '_rgb' 로 저장돼 상호 덮어쓰기).
    #    캐스케이드에서 실험을 가르는 축: 1차 모델 · 2차 모델 · 1차 채널 · lr · τ 선택 규칙 ·
    #    균형화 · 방어 · **조기종료(보조 가중치·임계값)**.
    #    방어 축이 빠지면 방어 캐스케이드가 기존(방어 없음) 결과를 덮어쓴다 — docs/10 §0.1.
    #    같은 이유로 조기종료 축이 빠지면 M4 결과가 Phase 10 의 기준선을 덮어쓴다.
    bal = "_bal" if args.balance else ""
    s2 = "" if args.stage2 == "charcnn" else f"-{args.stage2}"  # 기본 조합은 접미사 없이
    ch_tag = data_image._channel_suffix(args.channels, encoders)
    lr_tag = "" if args.lr == DEFAULT_LR else f"_lr{args.lr:g}"
    sel_tag = "" if args.select == "match-teacher" else f"_b{args.budget:g}"
    def_tag = defense_suffix(args.defense, args.mutation_split, args.aug_ratio)
    # 조기종료: 모델 축(_ee) + 학습 축(_aw) + 운영점 축(_ex, 직접 지정한 경우만).
    # 임계값을 val 에서 고른 기본 경로는 접미사 없이 둬 파일명이 짧게 유지된다(τ 관습 계승).
    ee_tag = ""
    if args.stage1 == "cnn_ee":
        ee_tag = ("_ee" + aux_weight_suffix(args.aux_weight)
                  + exit_suffix(args.exit_threshold))
    tag = f"{args.track}_cascade{s2}_{args.text}{ch_tag}{lr_tag}{sel_tag}{ee_tag}{def_tag}{bal}"
    M.save_report(result, RESULTS_DIR / f"{tag}.json")
    M.save_predictions(y_te, pred_te, classes, RESULTS_DIR / f"pred_{tag}.npz", y_score=prob_te)
    M.save_confusion_matrix(y_te, pred_te, classes, FIG_DIR / f"cm_{tag}.png",
                            title=f"cascade {args.stage1}->{args.stage2} ({args.track})")
    fig_pareto(test_rows, {"tau": tau, "escalation_rate": esc_rate,
                           "macro_f1": result["macro_f1"]}, oracle, ms1, ms2,
               FIG_DIR / f"cascade_curve_{tag}.png",
               f"Hybrid cascade: {args.stage1} (stage-1) -> {args.stage2} (stage-2), {args.track}")
    print(f"\n  [저장] 지표 → experiments/results/{tag}.json")
    print(f"  [저장] 예측 → experiments/results/pred_{tag}.npz")
    print(f"  [저장] 그림 → docs/figures/models/cascade_curve_{tag}.png, cm_{tag}.png")


if __name__ == "__main__":
    main()
