"""
Phase 5 (RQ2) — 회피 공격 실행 & ASR 산출 (CLI)

무엇을 하나:
    1) 대상 탐지기를 준비한다.
         · TF-IDF(logreg/rf): clean train 으로 즉석 학습(CPU 가능, 체크포인트 불필요).
         · torch(cnn/charcnn/bilstm): Phase 4 방식으로 미리 학습된 체크포인트를 로드(GPU 권장).
    2) problem_space 의 의미보존 변형을 test 공격 샘플에 적용해, **같은 전처리→예측 경로**로
       통과시킨다(입력만 오염, 파이프라인 동일 — docs/05 §6). 즉 변형된 텍스트를
       그 자리에서 이미지/바이트시퀀스로 바꿔 학습된 모델에 넣는다(파일 재빌드 없음).
    3) 회피 성공률(ASR)을 benign-evasion(공격→Normal 예측) 기준으로 측정한다.
         · 단일 기법별 ASR (어떤 회피가 잘 통하는가)
         · 예산(k) 조합 ASR 곡선 (회피를 겹칠수록 얼마나 뚫리나)

지표 정의(docs/05 §2.2) — 둘을 반드시 함께 보고한다:
    주 지표 = benign-evasion = 변형된 공격이 Normal 로 예측된 비율(= WAF 우회, 유일하게 위험한 실패).
    보조 지표 = any-misclassification = 예측≠진짜 클래스 비율(공격 클래스 사이 '동요').
    주 지표가 0 이어도 보조 지표로 강건성 차이가 드러난다.

핵심 설계 — 예측 인터페이스 추상화:
    모델 종류(TF-IDF / 이미지 CNN / 바이트 시퀀스)가 달라도 회피 로직은 동일해야 공정하다.
    그래서 각 모델을 `predict(list[str]) -> np.ndarray(클래스인덱스)` 라는 **하나의 콜러블**로
    감싼다. run_single/run_stacked 는 이 콜러블만 호출하므로 "같은 공격, 모델만 교체" 가 성립한다.

산출물:
    experiments/results/evasion_{track}_{model}_single.json / _stacked.json
    docs/figures/attacks/evasion_single_{track}_{model}.png (기법별 ASR)
    docs/figures/attacks/evasion_stacked_{track}_{model}.png (예산-ASR 곡선)
    (model 예: tfidf_logreg, tfidf_rf, cnn, charcnn, bilstm)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("src/models", "src/eval", "src/attacks", "src/imaging", "src/data"):
    sys.path.insert(0, str(PROJECT_ROOT / extra))

from data_text import (  # noqa: E402
    load_text_split, build_label_encoding, encode_byte_matrix,
)
from baseline_tfidf import build_classifier  # noqa: E402
from metrics import _find_normal_index  # noqa: E402
from payload_to_image import payload_to_image, payload_to_rgb_image  # noqa: E402
from preprocess import normalize_text  # noqa: E402  (입력 정규화 방어 — 전처리와 같은 함수를 쓴다)
import data_image  # noqa: E402  (채널 접미사 규칙 재사용)
from tagging import (  # noqa: E402
    DEFAULT_AUG_RATIO, DEFENSE_MODES, build_tag, defense_suffix,
)
import problem_space as PS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "attacks"
CKPT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"

# 모델 종류 구분: TF-IDF 계열은 즉석 학습, torch 계열은 체크포인트 로드.
TFIDF_MODELS = {"tfidf_logreg": "logreg", "tfidf_rf": "rf"}
TORCH_IMAGE_MODELS = {"cnn"}          # 변형 텍스트 → 이미지로 예측
TORCH_SEQ_MODELS = {"charcnn", "bilstm"}  # 변형 텍스트 → 바이트 시퀀스로 예측
# 하이브리드 캐스케이드(RGB CNN 1차 + char-CNN 2차, src/models/cascade.py).
# 단독 모델들과 "같은 공격·같은 파이프라인"으로 비교하려고 여기서도 대상에 포함한다.
CASCADE_MODELS = {"cascade": "charcnn", "cascade-bilstm": "bilstm"}
ALL_MODELS = (list(TFIDF_MODELS) + sorted(TORCH_IMAGE_MODELS | TORCH_SEQ_MODELS)
              + sorted(CASCADE_MODELS))


# ---------------------------------------------------------------------------
# 대상 모델 → 예측 콜러블 만들기 (predict: list[str] -> np.ndarray[int])
# ---------------------------------------------------------------------------
def apply_input_defense(texts, text_mode: str):
    """입력 정규화 방어(Phase 11 방어 B): 예측 **직전** 에 URL/HTML 디코딩을 적용한다.

    왜 여기인가: 공격자는 raw 트래픽에 변형을 가하고(docs/05 §3), 방어자는 탐지 전에 정규화한다.
    즉 '변형 → 정규화 → 탐지' 순서가 실제 배포 구조다. 전처리와 **같은 함수**(preprocess.normalize_text)
    를 쓰므로 학습(text_decoded 컬럼)과 추론의 정규화가 어긋날 수 없다.
    text_mode='raw' 면 아무것도 하지 않는다(기존 동작 그대로).
    """
    if text_mode != "decoded":
        return list(texts)
    return [normalize_text(t) for t in texts]


def build_tfidf_predict(track: str, clf_name: str, max_features: int, text: str = "raw"):
    """clean train 으로 TF-IDF+분류기를 학습해 (predict, classes) 를 반환한다.

    TF-IDF 는 gradient 가 필요 없고 학습이 가벼워, 체크포인트 대신 매 실행마다 clean train 으로
    즉석 학습한다(재현성: 같은 데이터·시드라 결과 동일). 불균형은 class_weight='balanced' 로
    보정하므로 별도 언더샘플링이 필요 없다(docs/05 §11-C).
    text='decoded' 면 정규화된 컬럼으로 학습하고 추론 입력도 같은 규칙으로 정규화한다(방어 B).
    """
    tr_txt, tr_lab = load_text_split(track, "train", text)
    y_train, classes = build_label_encoding(tr_lab)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          max_features=max_features, lowercase=False)
    Xtr = vec.fit_transform(tr_txt)
    clf = build_classifier(clf_name, len(classes))
    clf.fit(Xtr, y_train)

    def predict(texts):
        return clf.predict(vec.transform(apply_input_defense(texts, text)))

    return predict, classes


def build_torch_proba(track: str, model_name: str, side: int, max_len: int,
                      balanced: bool, device, text: str = "raw",
                      channels: str = "gray", encoders: tuple[str, ...] | None = None,
                      defense: str = "none", mutation_split: str | None = None,
                      aug_ratio: float | None = None):
    """미리 학습된 torch 체크포인트를 로드해 (proba, classes) 를 반환한다.

    proba: list[str] -> np.ndarray (N, K) softmax 확률.
    argmax 만 필요한 기존 경로(build_torch_predict)와, 확신도가 필요한 캐스케이드 경로가
    **같은 로딩·전처리 코드**를 공유하도록 확률 반환을 원본 함수로 둔다(분기 지점 단일화).

    체크포인트는 Phase 4 train.py 규칙으로 저장된 것을 그대로 쓴다:
        experiments/checkpoints/{track}_{model}_raw[_bal].pt
    RQ2 대상 모델은 --balance 로 학습된 '_bal' 체크포인트가 기본이다(docs/05 §11-C:
    Normal 이 실제 예측 후보가 되어야 benign-evasion 측정이 유효).

    클래스 순서는 train 라벨에서 build_label_encoding(sorted unique)으로 복원한다.
    이미지 트랙(build_image_dataset)과 텍스트 트랙(data_text)이 같은 규칙을 쓰므로,
    체크포인트가 학습된 라벨 인덱스와 정확히 일치한다.
    """
    import torch

    _, tr_lab = load_text_split(track, "train", text)
    _, classes = build_label_encoding(tr_lab)

    is_image = model_name in TORCH_IMAGE_MODELS
    in_channels = 3 if (is_image and channels == "rgb") else 1
    if is_image:
        import cnn as cnn_mod
        net = cnn_mod.build_model(len(classes), in_channels=in_channels)
    else:
        import text_models
        net = text_models.build_model(model_name, len(classes))

    # 체크포인트 이름은 train.py 와 **같은 규칙**(tagging.build_tag)으로 만든다.
    # 채널·방어 축이 빠지면 엉뚱한 가중치를 조용히 로드하게 된다.
    ckpt_tag = build_tag(track, model_name, text,
                         channels=channels if is_image else "gray",
                         encoders=encoders if is_image else None,
                         balance=balanced, defense=defense,
                         mutation_split=mutation_split, aug_ratio=aug_ratio)
    ckpt = CKPT_DIR / f"{ckpt_tag}.pt"
    if not ckpt.exists():
        retrain = (f"python src/models/train.py --model {model_name} --track {track}"
                   f"{' --balance' if balanced else ''}"
                   f"{f' --text {text}' if text != 'raw' else ''}"
                   f"{f' --channels {channels}' if channels != 'gray' else ''}"
                   f"{f' --defense {defense}' if defense != 'none' else ''}"
                   f"{f' --mutation-split {mutation_split} --aug-ratio {aug_ratio:g}' if defense == 'advtrain' else ''}")
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt}\n먼저 학습하세요: {retrain}")
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.to(device).eval()

    # 배치 크기: BiLSTM 은 긴 시퀀스(max_len=2304)를 순환 처리해 활성값 메모리가
    # 배치×길이에 비례해 폭증한다(1024 배치에서 30GiB+ OOM 발생). 그래서 bilstm 만
    # 작은 배치로 예측한다. conv 계열(cnn/charcnn)은 메모리가 가벼워 큰 배치가 안전·빠르다.
    default_batch = 128 if model_name == "bilstm" else 1024

    def proba(texts, batch: int = default_batch):
        """변형 텍스트를 이미지/바이트행렬로 그 자리에서 바꿔 배치 확률 추론한다.

        text='decoded' 면 변환 **전에** 정규화를 적용한다(= 입력 정규화 방어의 실제 배포 순서).
        """
        texts = apply_input_defense(texts, text)
        if not texts:
            return np.zeros((0, len(classes)), dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                if is_image and in_channels == 3:
                    # 텍스트 → (side,side,3) uint8 → (B,3,side,side) float(0~1)
                    # 채널 인코더 조합은 학습 때와 같아야 한다(tag 로 강제됨).
                    imgs = np.stack([payload_to_rgb_image(t, side, tuple(encoders)) for t in chunk])
                    x = torch.from_numpy(imgs.astype(np.float32) / 255.0)
                    x = x.permute(0, 3, 1, 2).contiguous().to(device)
                elif is_image:
                    # 텍스트 → (side,side) uint8 이미지 → (B,1,side,side) float(0~1)
                    imgs = np.stack([payload_to_image(t, side) for t in chunk])
                    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).unsqueeze(1).to(device)
                else:
                    # 텍스트 → (B, max_len) 바이트 인덱스 시퀀스
                    X = encode_byte_matrix(chunk, max_len)
                    x = torch.from_numpy(X).to(device)
                out.append(torch.softmax(net(x), dim=1).cpu().numpy())
        return np.concatenate(out)

    return proba, classes


def build_torch_predict(track: str, model_name: str, side: int, max_len: int,
                        balanced: bool, device, **kwargs):
    """단일 torch 모델을 `predict(list[str]) -> 클래스인덱스` 콜러블로 감싼다.

    kwargs(text/channels/encoders/defense/...)는 build_torch_proba 로 그대로 넘어간다 —
    분기 지점을 늘리지 않기 위해 여기서는 해석하지 않는다.
    """
    proba, classes = build_torch_proba(track, model_name, side, max_len, balanced, device, **kwargs)

    def predict(texts):
        return proba(texts).argmax(axis=1)

    return predict, classes


def load_cascade_tau(track: str, stage2: str, text: str, balanced: bool,
                     channels: str = "gray", encoders: tuple[str, ...] | None = None,
                     defense: str = "none", mutation_split: str | None = None,
                     aug_ratio: float | None = None) -> float:
    """cascade.py 가 val 에서 확정해 저장한 운영 임계값 τ 를 읽어온다.

    왜 파일에서 읽나: τ 를 여기서 다시 고르면 **회피 실험 데이터로 τ 를 튜닝**하는 셈이라
    공정성이 깨진다. 캐스케이드의 τ 는 clean val 에서 한 번 정해진 값이어야 하고,
    공격 실험은 그 고정된 운영점을 그대로 시험해야 한다.

    ⚠️ 찾는 파일의 이름은 **캐스케이드 구성과 정확히 같은 축**으로 만들어야 한다. 예전에는
    채널·방어 축이 빠져 있어 gray 구성의 τ 를 RGB 구성에 쓸 수 있었다(그래서 그 조합 자체를
    막아뒀다). 이제 cascade.py 가 축을 전부 tag 에 넣으므로 여기서도 같은 규칙으로 찾는다.
    다른 운영점을 시험하려면 `--tau` 로 직접 넘긴다.
    """
    s2 = "" if stage2 == "charcnn" else f"-{stage2}"
    ch_tag = data_image._channel_suffix(channels, encoders)
    def_tag = defense_suffix(defense, mutation_split, aug_ratio)
    path = (RESULTS_DIR /
            f"{track}_cascade{s2}_{text}{ch_tag}{def_tag}{'_bal' if balanced else ''}.json")
    if not path.exists():
        rebuild = (f"python src/models/cascade.py --track {track} --stage2 {stage2}"
                   f"{' --balance' if balanced else ''}"
                   f"{f' --channels {channels}' if channels != 'gray' else ''}"
                   f"{f' --text {text}' if text != 'raw' else ''}"
                   f"{f' --defense {defense}' if defense != 'none' else ''}"
                   f"{f' --mutation-split {mutation_split} --aug-ratio {aug_ratio:g}' if defense == 'advtrain' else ''}")
        raise FileNotFoundError(
            f"캐스케이드 리포트가 없습니다: {path}\n"
            f"  → 먼저 실행하세요: {rebuild}\n"
            f"  (또는 --tau 로 임계값을 직접 지정)"
        )
    with open(path, encoding="utf-8") as f:
        return float(json.load(f)["selected_tau"])


def build_cascade_predict(track: str, stage2: str, side: int, max_len: int,
                          balanced: bool, device, tau: float, text: str = "raw",
                          channels: str = "gray", encoders: tuple[str, ...] | None = None,
                          defense: str = "none", mutation_split: str | None = None,
                          aug_ratio: float | None = None):
    """하이브리드 캐스케이드를 `predict(list[str]) -> 클래스인덱스` 콜러블로 감싼다.

    1차(RGB CNN)가 전부 판정하고, 확신도 < τ 인 샘플만 2차(char-CNN)로 넘긴다.
    2차는 **에스컬레이션된 부분집합에만** 실행해 실제 배포 동작과 비용 구조를 그대로 재현한다.

    채널·방어 축은 두 단계에 **같은 값**으로 넘긴다(2차는 텍스트 모델이라 채널 축이 없어
    build_torch_proba 안에서 무시된다). cascade.py 가 τ 를 고를 때의 구성과 일치해야
    운영점이 의미를 갖는다.
    """
    proba1, classes = build_torch_proba(track, "cnn", side, max_len, balanced, device,
                                        text=text, channels=channels, encoders=encoders,
                                        defense=defense, mutation_split=mutation_split,
                                        aug_ratio=aug_ratio)
    proba2, classes2 = build_torch_proba(track, stage2, side, max_len, balanced, device,
                                         text=text, defense=defense,
                                         mutation_split=mutation_split, aug_ratio=aug_ratio)
    if classes != classes2:
        raise ValueError(f"두 단계의 클래스 순서가 다릅니다: {classes} vs {classes2}")

    def predict(texts):
        texts = list(texts)
        p1 = proba1(texts)
        pred = p1.argmax(axis=1)
        escalate = np.where(p1.max(axis=1) < tau)[0]
        if len(escalate):
            p2 = proba2([texts[i] for i in escalate])
            pred[escalate] = p2.argmax(axis=1)
        # 호출마다 에스컬레이션 비율을 기록한다.
        # 왜: 캐스케이드의 비용은 '얼마나 2차로 넘어가느냐'에 비례한다. 회피 변형이 1차
        # 확신도를 흔들면 에스컬레이션이 늘어 **지연이 증가**하는데(비용 기반 공격 표면),
        # 이는 정확도 지표로는 절대 안 보이는 하이브리드 고유의 리스크다.
        predict.escalation_log.append(float(len(escalate) / max(len(texts), 1)))
        return pred

    predict.escalation_log = []
    return predict, classes


# ---------------------------------------------------------------------------
# 지표 (docs/05 §2.2)
# ---------------------------------------------------------------------------
def benign_evasion(pred: np.ndarray, mask: np.ndarray, normal_idx: int) -> float:
    """mask(적용 대상) 샘플 중 Normal 로 예측된 비율(= 주 지표 회피 성공률, docs/05 §2.2 ①)."""
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == normal_idx).mean())


def any_misclass(pred: np.ndarray, mask: np.ndarray, y_true: np.ndarray) -> float:
    """mask 샘플 중 '예측 != 진짜 클래스' 비율(= 보조 지표, docs/05 §2.2 ②).

    주 지표(benign-evasion)가 0 이어도 탐지기가 공격 클래스 사이에서 얼마나 흔들리는지
    보여준다. WAF 우회는 아니지만 표현방식별 강건성 비교의 핵심 렌즈다.
    """
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] != y_true[mask]).mean())


def attach_escalation_rates(predict, single_rows, stacked_rows) -> bool:
    """캐스케이드일 때, 각 실험 행에 그때의 에스컬레이션 비율(2차 호출 비중)을 붙인다.

    왜 필요한가: 하이브리드의 비용은 '얼마나 2차로 넘어갔는가'로 결정된다. 회피 변형이
    1차 확신도를 흔들면 에스컬레이션이 늘어 **정확도는 그대로인데 지연만 커지는** 실패가
    가능하다(비용 기반 공격 표면). 정확도 지표만으로는 안 보이므로 함께 기록한다.

    호출 순서가 곧 로그 순서다(main 의 실행 순서에 의존):
        [0] clean 기준 예측 → [1 : 1+len(single)] 단일 기법 → 그 뒤 예산 k=1..budget.
    캐스케이드가 아니면(로그 속성 없음) 아무것도 하지 않고 False 를 반환한다.
    """
    log = getattr(predict, "escalation_log", None)
    if not log:
        return False
    for row, rate in zip(single_rows, log[1:1 + len(single_rows)]):
        row["escalation_rate"] = rate
    stacked_rows[0]["escalation_rate"] = log[0]  # stacked 의 k=0 행은 clean 기준점
    for row, rate in zip(stacked_rows[1:], log[1 + len(single_rows):]):
        row["escalation_rate"] = rate
    return True


def run_single(predict, te_txt, te_lab, y_true, base_pred, normal_idx, seed):
    """단일 기법별 ASR 을 계산한다. 반환: rows(list of dict)."""
    rows = []
    for tech in PS.ALL_TECHNIQUES:
        mutated, applied = PS.mutate_single(te_txt, te_lab, tech, random.Random(seed))
        pred = predict(mutated)
        rows.append({
            "technique": tech,
            "n_applied": int(applied.sum()),
            "asr_clean": benign_evasion(base_pred, applied, normal_idx),   # 주 지표 변형 전
            "asr_mutated": benign_evasion(pred, applied, normal_idx),      # 주 지표 변형 후
            # 보조 지표(any-misclassification) — 변형 전/후를 함께 기록해 '동요'를 정량화
            "anymis_clean": any_misclass(base_pred, applied, y_true),
            "anymis_mutated": any_misclass(pred, applied, y_true),
        })
        rows[-1]["asr_delta"] = rows[-1]["asr_mutated"] - rows[-1]["asr_clean"]
        rows[-1]["anymis_delta"] = rows[-1]["anymis_mutated"] - rows[-1]["anymis_clean"]
    return rows


def run_stacked(predict, te_txt, te_lab, y_true, base_pred, normal_idx, budget, seed):
    """예산 k=1..budget 조합 ASR 곡선을 계산한다(주·보조 지표 동시)."""
    attack_mask = np.asarray([bool(l in PS.ATTACK_LABELS) for l in te_lab])
    rows = []
    for k in range(1, budget + 1):
        mutated, applied = PS.mutate_stacked(te_txt, te_lab, k, random.Random(seed))
        pred = predict(mutated)
        rows.append({
            "budget_k": k,
            "n_applied": int(applied.sum()),
            "asr_mutated": benign_evasion(pred, applied, normal_idx),
            "anymis_mutated": any_misclass(pred, applied, y_true),
        })
    # k=0(=clean) 기준점도 앞에 붙여 곡선이 baseline 에서 출발하게 한다.
    rows.insert(0, {"budget_k": 0, "n_applied": int(attack_mask.sum()),
                    "asr_mutated": benign_evasion(base_pred, attack_mask, normal_idx),
                    "anymis_mutated": any_misclass(base_pred, attack_mask, y_true)})
    return rows


def fig_single(rows, out_path: Path, title: str, clean_ref: float):
    """기법별 ASR 가로 막대(변형 후), clean 기준선 표시."""
    rows_sorted = sorted(rows, key=lambda r: r["asr_mutated"])
    names = [r["technique"] for r in rows_sorted]
    asr = [r["asr_mutated"] for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, asr, color="#c0392b")
    ax.axvline(clean_ref, ls="--", color="gray", lw=1, label=f"clean baseline = {clean_ref:.3f}")
    ax.set_xlabel("ASR (benign-evasion: attack predicted as Normal)")
    ax.set_xlim(0, max(0.05, max(asr) * 1.15))
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


def fig_stacked(rows, out_path: Path, title: str):
    """예산 k vs 지표 곡선 — 주 지표(benign-evasion)와 보조 지표(any-misclass)를 함께."""
    x = [r["budget_k"] for r in rows]
    y_be = [r["asr_mutated"] for r in rows]
    y_am = [r["anymis_mutated"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y_be, "o-", color="#c0392b", lw=2, label="benign-evasion (primary: attack->Normal)")
    ax.plot(x, y_am, "s--", color="#2980b9", lw=2, label="any-misclass (secondary: pred!=true)")
    ax.set_xlabel("stacked mutation budget k")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(x)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


def get_device():
    """torch 계열일 때만 호출된다. GPU 있으면 cuda, 없으면 cpu."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description="RQ2 회피 공격 ASR 산출(TF-IDF/torch 대상 공용)")
    p.add_argument("--model", default="cnn", choices=ALL_MODELS,
                   help="공격 대상 모델. tfidf_* 는 즉석 학습, cnn/charcnn/bilstm 은 체크포인트 로드")
    p.add_argument("--track", default="payload_4class_csicnorm",
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"])
    p.add_argument("--budget", type=int, default=5, help="조합 공격 최대 예산 k")
    p.add_argument("--max-features", type=int, default=20000, help="TF-IDF 전용")
    p.add_argument("--side", type=int, default=48, help="이미지 한 변(cnn 전용)")
    p.add_argument("--max-len", type=int, default=48 * 48,
                   help="바이트 시퀀스 길이(charcnn/bilstm 전용). 기본=이미지 용량(48x48)과 동일")
    p.add_argument("--unbalanced", action="store_true",
                   help="torch 체크포인트를 '_bal' 없이(불균형 학습본) 로드. 기본은 balanced(_bal)")
    p.add_argument("--tau", type=float, default=None,
                   help="캐스케이드 임계값 직접 지정(cascade 전용). 기본은 cascade.py 가 "
                        "val 에서 확정해 저장한 값을 사용")
    p.add_argument("--seed", type=int, default=42)
    # ── Phase 11 (RQ3) — 방어 arm 을 같은 공격으로 재평가하기 위한 옵션 ────────
    p.add_argument("--text", default="raw", choices=["raw", "decoded"],
                   help="decoded = 입력 정규화 방어(방어 B). 변형 페이로드를 예측 직전에 "
                        "URL/HTML 디코딩한 뒤 탐지한다(학습도 --text decoded 본이어야 함)")
    p.add_argument("--channels", default="gray", choices=["gray", "rgb"],
                   help="이미지 모델 채널(cnn 전용). rgb 는 같은 조합으로 학습된 체크포인트가 필요")
    p.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy",
                   help="rgb 채널 인코더 R,G,B(학습 때와 동일해야 함)")
    p.add_argument("--defense", default="none", choices=list(DEFENSE_MODES),
                   help="공격 대상이 어떤 방어 모델인지(체크포인트 선택 + 산출물 tag). "
                        "advtrain 은 --mutation-split/--aug-ratio 로 어떤 방어본인지 지정")
    p.add_argument("--mutation-split", default="S0", choices=["S0", "SA"],
                   help="advtrain 체크포인트의 변형 분할(학습 때 쓴 값과 일치해야 함)")
    p.add_argument("--aug-ratio", type=float, default=DEFAULT_AUG_RATIO,
                   help="advtrain 체크포인트의 증강 비율(학습 때 쓴 값과 일치해야 함)")
    args = p.parse_args()

    # 캐스케이드는 운영점 τ 가 **그 구성의 clean val** 에서 확정돼 있어야 한다. 이제 cascade.py 가
    # 채널·방어 축을 tag 에 넣고 τ 를 따로 고르므로(docs/10 §0.1), 구성별 τ 를 정확히 찾아 쓸 수 있다.
    # 여기서 τ 를 다시 고르지 않는다는 원칙은 그대로다 — 공격 데이터로 운영점을 튜닝하면 누수다.

    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    print(f"=== RQ2 회피 ASR: track={args.track} model={args.model} "
          f"text={args.text} channels={args.channels} defense={args.defense} ===")
    t0 = time.perf_counter()

    # 대상 모델을 predict 콜러블로 감싼다(모델 종류를 이 지점에서만 분기).
    # is_cascade 는 저장 tag(운영점 τ 축)에서도 쓰이므로 분기 밖에서 한 번만 정한다.
    is_cascade = args.model in CASCADE_MODELS
    if args.model in TFIDF_MODELS:
        predict, classes = build_tfidf_predict(args.track, TFIDF_MODELS[args.model],
                                               args.max_features, text=args.text)
        device_note = "cpu(tfidf)"
    elif is_cascade:
        device = get_device()
        stage2 = CASCADE_MODELS[args.model]
        balanced = not args.unbalanced
        tau = args.tau if args.tau is not None else load_cascade_tau(
            args.track, stage2, args.text, balanced, args.channels, encoders,
            args.defense, args.mutation_split, args.aug_ratio)
        predict, classes = build_cascade_predict(
            args.track, stage2, args.side, args.max_len, balanced, device, tau,
            text=args.text, channels=args.channels, encoders=encoders,
            defense=args.defense, mutation_split=args.mutation_split,
            aug_ratio=args.aug_ratio)
        device_note = f"{device} / cascade(cnn->{stage2}, tau={tau:.4f}, {args.channels})" + \
                      ("" if args.unbalanced else " / bal")
    else:
        device = get_device()
        predict, classes = build_torch_predict(
            args.track, args.model, args.side, args.max_len,
            balanced=not args.unbalanced, device=device,
            text=args.text, channels=args.channels, encoders=encoders,
            defense=args.defense, mutation_split=args.mutation_split,
            aug_ratio=args.aug_ratio,
        )
        device_note = str(device) + ("" if args.unbalanced else " / bal")

    normal_idx = _find_normal_index(classes)
    print(f"  대상 준비 완료({time.perf_counter()-t0:.1f}s) "
          f"[{device_note}] classes={classes} normal_idx={normal_idx}")

    # ⚠️ 공격 입력은 항상 raw 다. 정규화 방어(--text decoded)는 '탐지 직전'에 적용되지
    #    '공격자가 정규화된 페이로드를 보낸다'는 뜻이 아니다(공격→정규화→탐지 순서).
    te_txt, te_lab = load_text_split(args.track, "test", "raw")
    te_lab = list(te_lab)
    # 진짜 클래스 인덱스(보조 지표 any-misclass 계산에 필요). classes 순서에 맞춘다.
    lab2idx = {c: i for i, c in enumerate(classes)}
    y_true = np.asarray([lab2idx[l] for l in te_lab])
    base_pred = predict(te_txt)

    # 단일 기법별 ASR (주: benign-evasion / 보조: any-misclass)
    single = run_single(predict, te_txt, te_lab, y_true, base_pred, normal_idx, args.seed)
    print("  [단일 기법별 — 주(benign-evasion) / 보조(any-misclass), 변형 후]")
    for r in sorted(single, key=lambda x: -x["anymis_mutated"]):
        print(f"    {r['technique']:<22} n={r['n_applied']:>6,}  "
              f"BE {r['asr_clean']:.4f}→{r['asr_mutated']:.4f} | "
              f"AM {r['anymis_clean']:.4f}→{r['anymis_mutated']:.4f} (Δ+{r['anymis_delta']:.4f})")

    # 예산 조합 곡선
    stacked = run_stacked(predict, te_txt, te_lab, y_true, base_pred, normal_idx,
                          args.budget, args.seed)
    print("  [예산 곡선 — benign-evasion / any-misclass]")
    for r in stacked:
        print(f"    k={r['budget_k']}  BE={r['asr_mutated']:.4f}  AM={r['anymis_mutated']:.4f}")

    # 캐스케이드 전용 — 변형이 '비용'에 주는 영향(에스컬레이션 비율 증가 = 지연 증가).
    if attach_escalation_rates(predict, single, stacked):
        print("  [캐스케이드 비용 — 에스컬레이션 비율(2차 호출 비중, 낮을수록 빠름)]")
        for r in stacked:
            print(f"    k={r['budget_k']}  escalation={r.get('escalation_rate', float('nan')):.4f}")

    # 저장
    # ⚠️ tag 는 실험을 가르는 축을 전부 담아야 한다(커밋 5eede2f 의 RGB ablation 덮어쓰기 사고).
    #    캐스케이드는 **운영점 τ 가 곧 다른 실험**이다 — 같은 모델이라도 τ 가 다르면
    #    정확도·에스컬레이션이 전혀 달라지므로, τ 를 직접 지정한 실행은 별도 파일로 남긴다.
    #    train.py 의 lr 규칙과 같은 관습: 기본 운영점(cascade.py 가 val 에서 확정한 τ)이면 생략.
    #    같은 이유로 Phase 11 의 방어 축(입력 정규화·채널·증강 방어본)도 tag 에 들어간다 —
    #    방어 arm 의 결과가 기준선 결과를 덮어쓰면 비교 자체가 불가능해진다.
    #    기본값(raw/gray/none)이면 접미사가 전부 비어 **기존 산출물 파일명과 그대로 호환**된다.
    tau_tag = f"_tau{args.tau:g}" if (is_cascade and args.tau is not None) else ""
    text_tag = "" if args.text == "raw" else f"_{args.text}"
    # ⚠️ 채널 축은 **캐스케이드에도** 붙여야 한다. 1차가 CNN 이므로 gray/RGB 는 서로 다른 실험이다.
    #    예전에는 `args.model == "cnn"` 일 때만 붙였는데, 당시엔 캐스케이드+채널 조합 자체가
    #    막혀 있어 드러나지 않았다. 차단을 푼 순간 RGB 캐스케이드 결과가 gray 결과를 덮어썼다
    #    (2026-08-02 실제 사고 — docs/10 §0.1 작업 중 발견).
    ch_tag = (data_image._channel_suffix(args.channels, encoders)
              if (args.model == "cnn" or is_cascade) else "")
    def_tag = defense_suffix(args.defense, args.mutation_split, args.aug_ratio)
    tag = f"{args.track}_{args.model}{text_tag}{ch_tag}{def_tag}{tau_tag}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"track": args.track, "model": args.model, "classes": classes,
            "text": args.text, "channels": args.channels,
            "defense": {"mode": args.defense,
                        "mutation_split": args.mutation_split if args.defense == "advtrain" else None,
                        "aug_ratio": args.aug_ratio if args.defense == "advtrain" else None}}
    if is_cascade:
        meta["tau"] = tau  # 어느 운영점의 결과인지 파일 안에서도 확인 가능하게
    with open(RESULTS_DIR / f"evasion_{tag}_single.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "results": single}, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / f"evasion_{tag}_stacked.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "budget": args.budget, "results": stacked},
                  f, ensure_ascii=False, indent=2)

    clean_ref = stacked[0]["asr_mutated"]  # k=0 기준(전체 공격 clean benign-evasion)
    fig_single(single, FIG_DIR / f"evasion_single_{tag}.png",
               f"RQ2 single-technique ASR — {args.model} ({args.track})", clean_ref)
    fig_stacked(stacked, FIG_DIR / f"evasion_stacked_{tag}.png",
                f"RQ2 stacked-budget ASR — {args.model} ({args.track})")
    print(f"\n[저장] 결과 → experiments/results/evasion_{tag}_*.json")
    print(f"[저장] 그림 → docs/figures/attacks/evasion_*_{tag}.png")


if __name__ == "__main__":
    main()
