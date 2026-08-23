"""
Phase 4 — 공용 분류 성능 지표 모듈 (RQ1 평가의 단일 진실 소스)

목적:
    RGB CNN 과 모든 베이스라인이 "똑같은 방식"으로 평가되도록, 지표 계산과
    리포트 저장을 이 한 파일에 모은다. 모델별로 지표를 따로 짜면 계산 방식이
    미묘하게 달라져 비교가 왜곡되므로, 평가 로직은 반드시 여기로 일원화한다.

설계 문서 6.1(RQ1) 요구 지표:
    - Accuracy, Macro-F1, 클래스별 Precision/Recall/F1
    - Confusion Matrix (그림)
    - ROC-AUC (One-vs-Rest, macro) — 확률/점수가 있을 때만
    - 추론 처리량(throughput) — 실제 WAF 배포 가능성 논의용

Phase 13 개편 (docs/13 §1.2, 2026-08-13 교수 미팅):
    ⚠️ **지표마다 근거 문헌이 다르다.** 예전 주석은 신설 4종을 전부 Arp et al. 권고로 묶어
    서술했으나, 원문 대조 결과 ECE 는 그 논문에 **나오지 않는다**(docs/13 §1.6). 심사에서
    원문을 열어보면 바로 드러나므로 아래처럼 지표별로 분리해 둔다.

    - 신설 `operating_points` : TPR@1%FPR · TPR@0.1%FPR · pAUC(FPR<=0.01)
        근거: Arp et al. P7 — "consider the curves only up to tractable false-positive
        rates and compute bounded AUC values". WAF 는 저 FPR 영역에서만 운영 가능하므로
        전 구간 평균인 ROC-AUC 는 실무적 차이를 가린다.
        pAUC 표준화 방식의 1차 출처는 McClish (1989) — PAUC_MAX_FPR 주석 참조.
    - 신설 `attack_focused.alert_load` : 배포 base rate 기준 경보 부하
        근거: Arp et al. P8(base rate fallacy). 그 계보의 원점은 보안 도메인의
        Axelsson (ACM CCS 1999) — DEPLOYMENT_ATTACK_PREVALENCES 주석 참조.
    - 신설 `calibration` : ECE(+MCE, 구간별 표)
        ⚠️ 근거는 Arp et al. 이 **아니다**. Naeini et al. (AAAI 2015, ECE 정의) 와
        Guo et al. (ICML 2017, reliability diagram + "현대 심층망은 더 이상 교정돼 있지
        않다") 이며, 캐스케이드 도메인 판본은 CalexNet (arXiv:2509.08318).
        필요성 논거는 calibration_metrics 독스트링에 적었다(과대 주장 정정됨).
    - 제거 `mcc` : 교수 지시로 완전 제거. 기준 지표는 Macro-F1 로 이관했다.
        ⚠️ 제거 근거와 반대 근거(Arp et al. 는 P8 권고 절에서 MCC 를 명시 권고한다 —
        원문 대조로 확인, docs/13 §1.6)는 docs/13 §1.1 에 보존.
        cv_compare 의 검정 기준과 cascade 의 임계값 선택 규칙도 함께 이관됨(docs/13 §1.4).

참고 문헌 (서지 정본은 docs/13 §6):
    Arp et al. USENIX Security 2022 / CACM 67(11):104-112, 2024. doi:10.1145/3643456
    Axelsson. The base-rate fallacy... ACM CCS 1999. doi:10.1145/319709.319710
    McClish. Analyzing a portion of the ROC curve. Med Decis Making 9(3):190-195, 1989.
    Naeini, Cooper, Hauskrecht. AAAI 2015 / Guo, Pleiss, Sun, Weinberger. ICML 2017.

의존성:
    torch 불필요 — numpy/scikit-learn/matplotlib 만 사용한다.
    (RGB CNN·베이스라인 어느 쪽이든 최종 예측(y_pred)/점수(y_score)만 넘기면 된다.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경(Colab/서버)에서도 그림 저장 가능하도록
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Phase 13 신설 지표의 기준값 (docs/13 §1.2)
# ---------------------------------------------------------------------------

# 운영 지점으로 보고할 FPR. WAF 는 오탐이 곧 정상 사용자 차단이라 1% 도 이미 후한 값이다.
# 두 지점을 함께 보고하는 이유: 배포 환경마다 감당 가능한 오탐률이 달라 하나로 못 정한다.
#
# ⚠️ **이 지표의 해상도는 test 셋의 Normal 표본 수가 결정한다** (docs/13 §1.7).
#    FPR 이 가질 수 있는 값은 (오탐 건수 / Normal 수) 의 배수뿐이므로, Normal 이 N 개면
#    해상도는 1/N 이다. 목표 FPR 이 1/N 보다 작으면 그 지점은 **존재하지 않고**, 코드는
#    "오탐 0건" 지점을 집어온다 → 사실상 "무오탐 탐지율"이 되고 표본 하나에 크게 흔들린다.
#    실측 예: payload_4class_csicnorm test 의 Normal 은 722개 → 해상도 0.139%
#             → TPR@1%FPR 은 계산 가능하지만 **TPR@0.1%FPR 은 측정 불가**.
#             SR-BH 2020 으로 교체하면 test Normal 이 약 78,779개가 되어 둘 다 성립한다.
#    → 저 FPR 지표를 보고할 때는 반드시 Normal 표본 수를 함께 싣는다.
TARGET_FPRS = (0.01, 0.001)

# pAUC 를 끊을 FPR 상한. Arp et al. P7: "감당 가능한 FPR 까지만 ROC 를 보고 bounded AUC 를 계산하라".
# 표준화(McClish 보정)의 1차 출처는 McClish, *Analyzing a portion of the ROC curve*,
# Medical Decision Making 9(3):190-195, 1989 — partial_roc_auc() 독스트링 참조.
# ⚠️ 위 TARGET_FPRS 의 해상도 제약이 여기에도 그대로 걸린다. [0, max_fpr] 구간의 ROC 곡선은
#    Normal 표본 중 이 구간에 들어오는 몇 개로만 그려지므로(722개 트랙에서는 약 7개),
#    표본이 적으면 pAUC 는 계단 몇 칸짜리 값이 되어 비교 근거로 못 쓴다.
PAUC_MAX_FPR = 0.01

# 경보 부하를 계산할 때 가정하는 **배포 환경의 공격 비율**(base rate).
# ⚠️ 우리 데이터셋의 공격 비율은 수집 방식 때문에 인위적으로 부풀려진 값이다. 트랙별 실측
#    (docs/03 분포 기준): payload_4class_csicnorm 96.8% · payload_4class 73.1% ·
#    csic_binary 64.1%. (SR-BH 2020 으로 교체하면 약 42% 가 된다 — 그때 이 주석을 갱신할 것.)
#    실제 웹 트래픽에서 공격은 극소수이므로, 그 비율로 환산하지 않으면 Precision 이
#    실제보다 훨씬 좋아 보인다(base rate fallacy). 정확한 값을 알 수 없으니 두 가정을 병기한다.
# 근거: Axelsson. *The base-rate fallacy and its implications for the difficulty of
#       intrusion detection.* ACM CCS 1999. doi:10.1145/319709.319710
#       — "침입 탐지의 한계 요인은 탐지율이 아니라 오탐률"임을 보인 이 계열의 원점.
DEPLOYMENT_ATTACK_PREVALENCES = (0.01, 0.001)

# 경보 부하를 표시할 기준 요청 수(= "100만 요청당 오탐 몇 건"). 사람이 읽을 수 있는 단위로 환산.
ALERT_LOAD_PER_REQUESTS = 1_000_000

# ECE 구간 수. 15 는 교정 오차 문헌의 관행값이며, 표본이 적을 때 구간이 비는 것을 피하는 절충점.
ECE_N_BINS = 15


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_score: np.ndarray | None = None,
) -> dict:
    """예측 결과로부터 RQ1 핵심 지표를 계산해 딕셔너리로 반환한다.

    인자:
        y_true   : (N,) 정수 정답 라벨
        y_pred   : (N,) 정수 예측 라벨
        class_names : 정수 코드 → 클래스명 (인덱스=코드)
        y_score  : (N, K) 클래스별 확률/점수. 있으면 점수 기반 지표(ROC-AUC·PR-AUC·
                   운영 지점·교정)를 추가로 계산. (일부 모델은 확률을 못 주므로 선택 인자.)

    반환: {accuracy, macro_f1, weighted_f1, per_class{...},
           roc_auc_ovr?, pr_auc_macro?, calibration?, operating_points?, attack_focused?}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    # 클래스별 precision/recall/f1/support (labels 를 명시해 클래스 순서를 고정)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
    }

    # 정상 클래스 위치를 먼저 찾는다 — 보안 지표와 운영 지점 계산이 모두 이 인덱스를 쓴다.
    normal_idx = _find_normal_index(class_names)

    # 점수 기반 지표: 확률/점수가 있을 때만.
    if y_score is not None:
        y_score = np.asarray(y_score)
        result["roc_auc_ovr"] = _safe_roc_auc(y_true, y_score, n_classes)
        # PR-AUC(Average Precision, macro): 다수 음성(TN) 상황에서 ROC보다 정보량이 큰 불균형 지표.
        result["pr_auc_macro"] = _safe_pr_auc(y_true, y_score, n_classes)
        # 교정(ECE): 확신도 자체의 신뢰도. 캐스케이드 tau 게이트의 전제(docs/13 §1.2).
        result["calibration"] = calibration_metrics(y_true, y_score)
        # 운영 지점(TPR@FPR, pAUC): "공격 vs 정상" 이진 접기 위에서만 정의된다.
        if normal_idx is not None:
            result["operating_points"] = attack_operating_points(y_true, y_score, normal_idx)

    # 보안 중심 지표: Normal 클래스가 있으면 "공격/정상" 관점으로 접어서 함께 계산한다.
    # (clean Macro-F1 은 다수 Normal·표면토큰 shortcut 때문에 포화되어 실제 탐지력을 가린다.
    #  RQ 관점에서 진짜 중요한 건 "공격을 공격으로 잡았는가 / 정상을 오탐했는가"이다.)
    if normal_idx is not None:
        result["attack_focused"] = attack_focused_metrics(y_true, y_pred, class_names, normal_idx)

    return result


def _find_normal_index(class_names: list[str]) -> int | None:
    """정상 클래스의 인덱스를 이름으로 찾는다(없으면 None).

    데이터셋마다 정상 라벨 표기가 다를 수 있어 대소문자 무시 부분일치로 관대하게 매칭한다.
    (예: "Normal", "normal", "benign", "valid")
    """
    for i, name in enumerate(class_names):
        if any(key in name.lower() for key in ("normal", "benign", "valid")):
            return i
    return None


def attack_score(y_score: np.ndarray, normal_idx: int) -> np.ndarray:
    """다중 클래스 확률을 "공격일 확률" 한 개로 접는다.

    왜 1 - P(Normal) 인가:
        WAF 의 실제 결정은 "막을지 말지"이므로, 어떤 공격 유형인지와 무관하게
        '정상이 아닐 확률'이 곧 운영 점수다. 공격 클래스 확률을 합치는 것과 수학적으로
        같지만(확률 합=1), 정상 클래스 하나만 보므로 클래스 수가 바뀌어도 식이 안 변한다.
    """
    y_score = np.asarray(y_score, dtype=float)
    return 1.0 - y_score[:, normal_idx]


def tpr_at_fpr(y_binary: np.ndarray, score: np.ndarray, target_fpr: float) -> dict:
    """FPR 을 target 이하로 묶었을 때 달성 가능한 TPR(과 그 임계값)을 구한다.

    왜 필요한가 (Arp et al., docs/13 §1.2):
        ROC-AUC 는 FPR 0~1 전 구간의 평균이라, 실무에서 절대 쓰지 않는 고오탐 구간까지
        점수에 넣는다. WAF 운영자가 실제로 묻는 것은 "오탐 1% 를 감수하면 공격을 몇 % 잡나"다.

    계산 방식:
        보간하지 않고 **FPR <= target 인 지점 중 가장 오른쪽**(TPR 최대)을 고른다.
        보간값은 실제로 달성 가능한 운영점이 아니므로 보고용으로 부적절하다.

    반환: {target_fpr, tpr, threshold, achieved_fpr} — 계산 불가 시 각 값이 None.
    """
    y_binary = np.asarray(y_binary)
    # 한쪽 클래스만 있으면 ROC 가 정의되지 않는다(예: 정상 표본이 없는 외부 검증셋).
    if y_binary.min() == y_binary.max():
        return {"target_fpr": target_fpr, "tpr": None, "threshold": None, "achieved_fpr": None}

    fprs, tprs, thresholds = roc_curve(y_binary, score)
    ok = np.nonzero(fprs <= target_fpr)[0]
    if ok.size == 0:  # roc_curve 는 항상 (0,0) 을 포함하므로 사실상 발생하지 않지만 방어.
        return {"target_fpr": target_fpr, "tpr": None, "threshold": None, "achieved_fpr": None}

    i = ok[-1]
    return {
        "target_fpr": target_fpr,
        "tpr": float(tprs[i]),
        "threshold": float(thresholds[i]),
        "achieved_fpr": float(fprs[i]),
    }


def partial_roc_auc(y_binary: np.ndarray, score: np.ndarray,
                    max_fpr: float = PAUC_MAX_FPR) -> float | None:
    """FPR <= max_fpr 구간만 본 ROC-AUC(pAUC)를 반환한다.

    sklearn 의 `max_fpr` 은 McClish 보정(McClish, Med Decis Making 9(3):190-195, 1989)을
    적용해 값을 [0.5, 1] 로 표준화한다.
    즉 0.5 = 무작위, 1.0 = 완벽이며 **전 구간 ROC-AUC 와 직접 비교하면 안 된다**
    (같은 척도가 아니다). 논문 표에서는 pAUC 열을 따로 두고 상한 FPR 을 함께 명시할 것.
    """
    y_binary = np.asarray(y_binary)
    # 한쪽 클래스만 있으면 ROC 가 정의되지 않는다(예: 정상 표본이 없는 외부 검증셋).
    # tpr_at_fpr 과 같은 방식으로 **먼저 명시적으로** 걸러 낸다 — sklearn 의 처리 방식에
    # 기대지 않기 위해서다(아래 _finite_or_none 주석 참조).
    if y_binary.size == 0 or y_binary.min() == y_binary.max():
        return None

    try:
        return _finite_or_none(roc_auc_score(y_binary, score, max_fpr=max_fpr))
    except ValueError:
        return None


def attack_operating_points(y_true: np.ndarray, y_score: np.ndarray, normal_idx: int) -> dict:
    """"공격 vs 정상" 이진 접기 위에서 저 FPR 운영 지점들을 계산한다.

    다중 클래스 문제여도 WAF 의 결정은 이진(차단/통과)이므로, 운영 지점은 이 접힌
    문제에서 정의하는 것이 맞다. 공격 유형 구분 성능은 Macro-F1 이 따로 담당한다.
    """
    y_true = np.asarray(y_true)
    is_attack = (y_true != normal_idx).astype(int)
    score = attack_score(y_score, normal_idx)

    return {
        "positive_class": "attack (not Normal)",
        "pauc_max_fpr": PAUC_MAX_FPR,
        "pauc": partial_roc_auc(is_attack, score, PAUC_MAX_FPR),
        "tpr_at_fpr": [tpr_at_fpr(is_attack, score, t) for t in TARGET_FPRS],
        "note": ("pAUC 는 McClish 보정으로 [0.5,1] 표준화된 값 — 전 구간 ROC-AUC 와 "
                 "같은 척도가 아니다."),
    }


def calibration_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        n_bins: int = ECE_N_BINS) -> dict:
    """확신도 교정 오차(ECE·MCE)와 reliability diagram 용 구간별 표를 계산한다.

    왜 필요한가 (⚠️ 2026-08-23 정정 — 이전 논거는 과대 주장이었다, docs/13 §1.6):
        예전 주석은 "확신도가 교정 안 되면 게이트가 승급 대상을 못 골라 캐스케이드의 전제가
        무너진다"고 적었으나 이는 정확하지 않다. tau 게이트에 필요한 것은 확신도의
        **순위(ranking)** 이지 수치의 교정이 아니다. 확신도가 전부 0.99 쪽에 쏠려 있어도
        val 에서 tau=0.995 로 잡히면 게이트는 정상 작동한다. 즉 "val 에서 tau 를 튜닝하는데
        교정이 왜 필요한가"라는 반문에 그 논거는 버티지 못한다.

        실제 값어치는 다음 셋이다:
        (ㄱ) tau 의 이식성 — 교정돼 있어야 val 에서 고른 tau 가 다른 데이터셋·배포 환경에서도
             같은 의미를 갖는다. Data 2025 외부셋 실험이 정확히 이 검증이다.
        (ㄴ) tau 의 해석 가능성 — "확신도 0.9 미만은 승급"이 실제로 "오답 확률 10%"를 뜻하는지.
        (ㄷ) **틀린 샘플에 과확신하는 실패의 진단** — 이건 순위 자체를 망가뜨리므로 게이트에
             실제 타격이다. 평균인 ECE 가 가리는 이 국소 실패를 MCE 가 드러낸다.

    정의:
        확신도 = max 확률, 정답 여부 = argmax 가 맞았는지. 확신도를 n_bins 개 등폭 구간으로
        나눠 |구간 정확도 - 구간 평균확신도| 를 표본 수로 가중 평균한 것이 ECE.
        MCE 는 그 최대값(최악 구간) — 평균이 가려버리는 국소 실패를 드러낸다.

    반환: {ece, mce, n_bins, bins[{lo,hi,count,avg_confidence,accuracy,gap}]}
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    confidence = y_score.max(axis=1)
    correct = (y_score.argmax(axis=1) == y_true).astype(float)
    n_total = len(y_true)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # 마지막 구간만 오른쪽 끝을 포함해야 확신도 1.0 인 샘플이 누락되지 않는다.
        in_bin = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "avg_confidence": None, "accuracy": None, "gap": None})
            continue
        avg_conf = float(confidence[in_bin].mean())
        acc = float(correct[in_bin].mean())
        gap = abs(acc - avg_conf)
        ece += (count / n_total) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": count,
                     "avg_confidence": avg_conf, "accuracy": acc, "gap": float(gap)})

    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins, "bins": bins}


def alert_load(tpr: float | None, fpr: float | None,
               prevalences: tuple[float, ...] = DEPLOYMENT_ATTACK_PREVALENCES,
               per_requests: int = ALERT_LOAD_PER_REQUESTS) -> list[dict]:
    """배포 환경의 공격 비율을 가정해 "실제로 사람이 처리할 경보량"을 환산한다.

    왜 필요한가 (base rate fallacy — Axelsson CCS 1999, Arp et al. P8):
        우리 test 셋의 공격 비율(트랙별 64~97%, DEPLOYMENT_ATTACK_PREVALENCES 주석의 실측치)은
        수집 방식 때문에 부풀려진 값이다. 그 비율에서 계산한 Precision 은 실배포에서 의미가 없다. 실제 웹 트래픽은 대부분 정상이므로,
        FPR 이 0.5% 라도 하루 수백만 요청에서는 오탐이 수천 건이 되어 운영이 불가능해진다.
        그 간극을 숫자로 보여 주는 것이 이 함수다.

    계산: 공격 비율 pi 에서
        진짜 경보 = TPR * pi * N,  헛 경보 = FPR * (1-pi) * N
        precision  = 진짜 / (진짜 + 헛)      ← base rate 를 반영한 정밀도

    예) FPR=0.002, TPR=0.99, pi=0.001, N=100만 → 헛 경보 1,998건 vs 진짜 990건
        → precision 0.331. 즉 경보 3건 중 2건이 헛것이다(원 데이터셋 기준으로는 0.99 로 보인다).
    """
    if tpr is None or fpr is None:
        return []

    rows = []
    for pi in prevalences:
        true_alerts = tpr * pi * per_requests
        false_alerts = fpr * (1.0 - pi) * per_requests
        total = true_alerts + false_alerts
        rows.append({
            "attack_prevalence": pi,
            "per_requests": per_requests,
            "true_alerts": float(true_alerts),
            "false_alerts": float(false_alerts),
            "precision_at_prevalence": float(true_alerts / total) if total > 0 else None,
        })
    return rows


def attack_focused_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normal_idx: int,
) -> dict:
    """다중 클래스를 "공격 vs 정상"으로 접어 보안적으로 의미 있는 지표를 계산한다.

    왜 필요한가:
        Macro-F1/Accuracy 는 다수인 Normal 과 표면 토큰 shortcut 덕에 쉽게 0.99 로 포화된다.
        하지만 WAF 관점에서 진짜 실패는 "공격이 Normal 로 새는 것(benign-evasion)"이다.
        이 지표는 그 실패를 정면으로 드러낸다.

    반환:
        attack_detection_recall : 진짜 공격 중 '공격(= Normal 이 아닌 어떤 클래스)'으로 잡은 비율
        benign_evasion_rate     : 진짜 공격이 Normal 로 예측된 비율(보안적으로 가장 위험한 실패)
        normal_false_positive_rate : 진짜 Normal 이 공격으로 오탐된 비율
        per_attack_leak : 공격 클래스별 Normal 누출률(어떤 공격이 가장 잘 새는지)
        n_attacks / n_normal : 표본 수(해석용)
        observed_attack_prevalence : 이 셋의 공격 비율(아래 alert_load 의 가정과 대조용)
        alert_load : 배포 base rate 를 가정한 경보 부하(Phase 13 신설, docs/13 §1.2)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    is_true_attack = y_true != normal_idx
    is_true_normal = ~is_true_attack
    pred_is_normal = y_pred == normal_idx

    n_attacks = int(is_true_attack.sum())
    n_normal = int(is_true_normal.sum())

    # 공격 표본에서: Normal 로 예측되면 회피(누출), 그 외는 '탐지됨'으로 본다.
    leaked = int((is_true_attack & pred_is_normal).sum())
    detected = n_attacks - leaked

    # 공격 클래스별 Normal 누출률(어느 공격 유형이 가장 잘 빠져나가는지 진단)
    per_attack_leak = {}
    for code, name in enumerate(class_names):
        if code == normal_idx:
            continue
        mask = y_true == code
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        leak = int((mask & pred_is_normal).sum())
        per_attack_leak[name] = {
            "leak_to_normal": leak,
            "support": cnt,
            "leak_rate": float(leak / cnt),
        }

    fp = int((is_true_normal & ~pred_is_normal).sum())

    recall = float(detected / n_attacks) if n_attacks else None
    fpr = float(fp / n_normal) if n_normal else None
    n_all = n_attacks + n_normal

    return {
        "normal_class": class_names[normal_idx],
        "n_attacks": n_attacks,
        "n_normal": n_normal,
        "attack_detection_recall": recall,
        "benign_evasion_rate": float(leaked / n_attacks) if n_attacks else None,
        "normal_false_positive_rate": fpr,
        "per_attack_leak": per_attack_leak,
        # 이 셋의 실제 공격 비율. alert_load 의 가정 공격 비율과 얼마나 다른지 보여 주기 위해
        # 함께 남긴다 — 이 격차가 곧 base rate fallacy 의 크기다.
        "observed_attack_prevalence": float(n_attacks / n_all) if n_all else None,
        "alert_load": alert_load(recall, fpr),
    }


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    y_score: np.ndarray | None = None,
) -> Path:
    """샘플 단위 예측을 .npz 로 저장한다(모델 간 '탐지 불일치' 분석의 재료).

    왜 필요한가:
        "정확도가 낮아지더라도 기존 방법이 못 잡던 걸 이미지 CNN 이 잡는가"를 보려면
        집계 지표가 아니라 '어느 샘플을' 각 모델이 맞췄는지가 필요하다. 여기서 남긴
        예측을 detection_analysis.py 가 test CSV(원문 페이로드)와 행 인덱스로 join 한다.

    저장 규약:
        - 행 인덱스(idx)는 test CSV 의 행 순서와 1:1 (로더가 순서를 보존하므로).
        - y_score 는 모델이 확률을 줄 때만 저장(없으면 생략).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    arrays = {
        "idx": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": np.asarray(class_names),
    }
    if y_score is not None:
        arrays["y_score"] = np.asarray(y_score)
    np.savez_compressed(out_path, **arrays)
    return out_path


def _finite_or_none(value) -> float | None:
    """nan/inf 를 None 으로 바꾼다 — 지표가 조용히 오염되는 것을 막는 마지막 관문.

    왜 필요한가 (2026-08-23 실제 사고, docs/13 §5):
        sklearn 은 "계산 불가"를 **판본에 따라 두 가지 방식**으로 알린다 —
        ValueError 를 던지거나, UndefinedMetricWarning 과 함께 **nan 을 돌려주거나**.
        우리 코드는 앞의 것만 잡고 있어서(`except ValueError`) 뒤의 경우 nan 이 그대로 통과했다.
        (`test_pauc_returns_none_when_roc_is_undefined` 가 이 기기에서 실패한 원인이다.)

        nan 이 새면 실패가 눈에 안 보이는 형태로 번진다:
        - cross_validate 의 fold 요약은 값이 수치형인지만 검사하므로 nan 이 통과 →
          평균 전체가 nan 이 되거나, cv_compare 의 순위 정렬이 조용히 틀어진다.
        - json.dump 는 nan 을 `NaN` 으로 쓰는데, 이는 표준 JSON 이 아니라 다른 도구가 못 읽는다.

        → 예외 처리와 **반환값 검사를 둘 다** 해야 한다. sklearn 판본을 고정하는 것으로는
          다른 기기(노트북/데스크톱)에서 재현되지 않는다.
    """
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> float | None:
    """ROC-AUC(OvR macro)를 계산하되, 계산 불가 상황에서는 None 을 돌려준다.

    계산 불가 예: test 셋에 특정 클래스가 하나도 없을 때 등.
    (평가 스크립트 전체가 죽지 않도록 여기서 방어적으로 감싼다.)
    """
    try:
        if n_classes == 2:
            # 이진 분류는 양성(코드 1) 클래스 점수 한 열만 사용한다.
            return _finite_or_none(roc_auc_score(y_true, y_score[:, 1]))
        return _finite_or_none(
            roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        )
    except ValueError:
        return None


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> float | None:
    """PR-AUC(Average Precision, OvR macro)를 계산하되 불가 시 None.

    불균형 보안 데이터에서 ROC-AUC 보완 지표(docs/07 리서치 근거). 이진은 양성 열만,
    다중 클래스는 각 클래스를 one-vs-rest 로 이진화해 macro 평균한다.
    """
    try:
        if n_classes == 2:
            return _finite_or_none(average_precision_score(y_true, y_score[:, 1]))
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        return _finite_or_none(
            average_precision_score(y_true_bin, y_score, average="macro")
        )
    except ValueError:
        return None


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Confusion Matrix 를 히트맵 그림(PNG)으로 저장한다(논문 그림용)."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 1.6 * len(class_names) + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # 각 칸에 개수를 적어 가독성을 높인다. 배경이 진하면 흰 글씨로 대비.
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in labels:
        for j in labels:
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_reliability_diagram(calibration: dict, out_path: Path,
                             title: str = "Reliability Diagram") -> Path | None:
    """교정 곡선(reliability diagram)을 PNG 로 저장한다(논문 그림용).

    읽는 법: 대각선이 완벽한 교정이다. 막대가 대각선보다 **아래**면 과신
    (0.9 라고 말하지만 실제 정확도는 0.7) — 캐스케이드 게이트가 승급 대상을 놓치는 방향이다.

    ⚠️ 라벨은 ASCII 만 쓴다(DejaVu Sans 에 한글 글리프가 없어 □ 로 깨짐, CLAUDE.md).
    """
    bins = [b for b in calibration.get("bins", []) if b.get("count")]
    if not bins:
        return None

    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["accuracy"] for b in bins]
    confs = [b["avg_confidence"] for b in bins]
    width = (bins[0]["hi"] - bins[0]["lo"]) * 0.9

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.bar(centers, accs, width=width, color="#4C78A8", edgecolor="white", label="accuracy")
    ax.plot(centers, confs, "o-", color="#E45756", ms=4, lw=1.2, label="avg confidence")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={calibration['ece']:.4f}  MCE={calibration['mce']:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_report(metrics: dict, out_path: Path) -> Path:
    """지표 딕셔너리를 JSON 으로 저장한다(모델 간 비교·논문 표 작성용)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return out_path


def format_summary(model_name: str, metrics: dict) -> str:
    """콘솔 출력용 한 줄 요약 문자열을 만든다."""
    parts = [
        f"[{model_name}]",
        f"acc={metrics['accuracy']:.4f}",
        f"macroF1={metrics['macro_f1']:.4f}",
    ]
    if metrics.get("pr_auc_macro") is not None:
        parts.append(f"PR-AUC={metrics['pr_auc_macro']:.4f}")
    # 운영 지점(Phase 13): 저 FPR 에서의 탐지율이 헤드라인이다. 첫 목표 FPR 만 한 줄에 노출.
    op = metrics.get("operating_points") or {}
    for row in (op.get("tpr_at_fpr") or [])[:1]:
        if row.get("tpr") is not None:
            parts.append(f"TPR@{row['target_fpr']:.1%}FPR={row['tpr']:.4f}")
    if op.get("pauc") is not None:
        parts.append(f"pAUC={op['pauc']:.4f}")
    cal = metrics.get("calibration") or {}
    if cal.get("ece") is not None:
        parts.append(f"ECE={cal['ece']:.4f}")
    if metrics.get("roc_auc_ovr") is not None:
        parts.append(f"AUC(OvR)={metrics['roc_auc_ovr']:.4f}")
    # 보안 지표가 있으면 "공격이 Normal 로 새는 비율"과 오탐률(FPR)을 함께 보여 준다(포화된 F1 보완).
    af = metrics.get("attack_focused")
    if af and af.get("benign_evasion_rate") is not None:
        parts.append(f"benign-evasion={af['benign_evasion_rate']:.4f}")
        if af.get("normal_false_positive_rate") is not None:
            parts.append(f"FPR={af['normal_false_positive_rate']:.4f}")
    return " ".join(parts)
