"""
Phase 4 — 공용 분류 성능 지표 모듈 (RQ1 평가의 단일 진실 소스)

목적:
    제안 CNN과 모든 베이스라인이 "똑같은 방식"으로 평가되도록, 지표 계산과
    리포트 저장을 이 한 파일에 모은다. 모델별로 지표를 따로 짜면 계산 방식이
    미묘하게 달라져 비교가 왜곡되므로, 평가 로직은 반드시 여기로 일원화한다.

설계 문서 6.1(RQ1) 요구 지표:
    - Accuracy, Macro-F1, 클래스별 Precision/Recall/F1
    - Confusion Matrix (그림)
    - ROC-AUC (One-vs-Rest, macro) — 확률/점수가 있을 때만
    - 추론 처리량(throughput) — 실제 WAF 배포 가능성 논의용

의존성:
    torch 불필요 — numpy/scikit-learn/matplotlib 만 사용한다.
    (제안 CNN·베이스라인 어느 쪽이든 최종 예측(y_pred)/점수(y_score)만 넘기면 된다.)
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
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


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
        y_score  : (N, K) 클래스별 확률/점수. 있으면 ROC-AUC(OvR)를 추가로 계산.
                   (일부 모델은 확률을 못 주므로 선택 인자로 둔다.)

    반환: {accuracy, macro_f1, weighted_f1, per_class{...}, roc_auc_ovr?}
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
        # MCC(Matthews 상관계수): 불균형 데이터에 강건한 단일 요약 지표.
        #   문헌 근거(docs/07): "높은 MCC는 항상 높은 ROC-AUC를 함의하나 역은 아님" → Acc/F1이
        #   다수·shortcut으로 포화될 때 실제 상관을 정직하게 보여준다. 논문 헤드라인 지표로 승격.
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": per_class,
    }

    # 점수 기반 지표(ROC-AUC, PR-AUC): 확률/점수가 있을 때만.
    if y_score is not None:
        y_score = np.asarray(y_score)
        result["roc_auc_ovr"] = _safe_roc_auc(y_true, y_score, n_classes)
        # PR-AUC(Average Precision, macro): 다수 음성(TN) 상황에서 ROC보다 정보량이 큰 불균형 지표.
        result["pr_auc_macro"] = _safe_pr_auc(y_true, y_score, n_classes)

    # 보안 중심 지표: Normal 클래스가 있으면 "공격/정상" 관점으로 접어서 함께 계산한다.
    # (clean Macro-F1 은 다수 Normal·표면토큰 shortcut 때문에 포화되어 실제 탐지력을 가린다.
    #  RQ 관점에서 진짜 중요한 건 "공격을 공격으로 잡았는가 / 정상을 오탐했는가"이다.)
    normal_idx = _find_normal_index(class_names)
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

    return {
        "normal_class": class_names[normal_idx],
        "n_attacks": n_attacks,
        "n_normal": n_normal,
        "attack_detection_recall": float(detected / n_attacks) if n_attacks else None,
        "benign_evasion_rate": float(leaked / n_attacks) if n_attacks else None,
        "normal_false_positive_rate": float(fp / n_normal) if n_normal else None,
        "per_attack_leak": per_attack_leak,
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


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> float | None:
    """ROC-AUC(OvR macro)를 계산하되, 계산 불가 상황에서는 None 을 돌려준다.

    계산 불가 예: test 셋에 특정 클래스가 하나도 없을 때 등.
    (평가 스크립트 전체가 죽지 않도록 여기서 방어적으로 감싼다.)
    """
    try:
        if n_classes == 2:
            # 이진 분류는 양성(코드 1) 클래스 점수 한 열만 사용한다.
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> float | None:
    """PR-AUC(Average Precision, OvR macro)를 계산하되 불가 시 None.

    불균형 보안 데이터에서 ROC-AUC 보완 지표(docs/07 리서치 근거). 이진은 양성 열만,
    다중 클래스는 각 클래스를 one-vs-rest 로 이진화해 macro 평균한다.
    """
    try:
        if n_classes == 2:
            return float(average_precision_score(y_true, y_score[:, 1]))
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        return float(average_precision_score(y_true_bin, y_score, average="macro"))
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
    # MCC: 불균형에 강건한 헤드라인 지표(docs/07). 항상 있으므로 함께 노출.
    if metrics.get("mcc") is not None:
        parts.append(f"MCC={metrics['mcc']:.4f}")
    if metrics.get("pr_auc_macro") is not None:
        parts.append(f"PR-AUC={metrics['pr_auc_macro']:.4f}")
    if metrics.get("roc_auc_ovr") is not None:
        parts.append(f"AUC(OvR)={metrics['roc_auc_ovr']:.4f}")
    # 보안 지표가 있으면 "공격이 Normal 로 새는 비율"과 오탐률(FPR)을 함께 보여 준다(포화된 F1 보완).
    af = metrics.get("attack_focused")
    if af and af.get("benign_evasion_rate") is not None:
        parts.append(f"benign-evasion={af['benign_evasion_rate']:.4f}")
        if af.get("normal_false_positive_rate") is not None:
            parts.append(f"FPR={af['normal_false_positive_rate']:.4f}")
    return " ".join(parts)
