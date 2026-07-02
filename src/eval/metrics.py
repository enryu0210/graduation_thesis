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
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


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
        "per_class": per_class,
    }

    # ROC-AUC(OvR macro): 점수가 있을 때만. 이진/다중 클래스 모두 처리.
    if y_score is not None:
        result["roc_auc_ovr"] = _safe_roc_auc(y_true, np.asarray(y_score), n_classes)

    return result


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
    if metrics.get("roc_auc_ovr") is not None:
        parts.append(f"AUC(OvR)={metrics['roc_auc_ovr']:.4f}")
    return " ".join(parts)
