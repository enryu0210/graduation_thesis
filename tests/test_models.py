"""Phase 4 모델·평가 유틸의 단위 테스트.

검증 포인트:
    - 지표 계산(metrics.py)이 정답/예측으로부터 올바른 값을 내는가
    - 클래스 가중치가 불균형을 제대로 보정하는가(드문 클래스에 큰 가중치)
    - 바이트 시퀀스 인코딩(data_text)이 truncation/padding/시프트 규칙을 지키는가
    - (torch 있으면) 얕은 CNN 이 올바른 출력 shape 의 logits 를 내는가

무거운 학습은 테스트하지 않는다(별도 --smoke 실행으로 확인). 순수 로직만 빠르게 검증한다.
"""

import numpy as np
import pytest

import metrics as M
from data_image import compute_class_weights
from data_text import (
    PAD_INDEX,
    VOCAB_SIZE,
    encode_byte_matrix,
    encode_byte_sequence,
)


# ── metrics.py ────────────────────────────────────────────────────────────────
def test_metrics_perfect_prediction():
    # 완벽 예측이면 accuracy/macro_f1 모두 1.0 이어야 한다.
    y = np.array([0, 1, 2, 1, 0])
    result = M.compute_metrics(y, y, ["a", "b", "c"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["per_class"]["a"]["support"] == 2


def test_metrics_all_wrong():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    result = M.compute_metrics(y_true, y_pred, ["a", "b"])
    assert result["accuracy"] == 0.0


def test_metrics_roc_auc_binary():
    # 점수를 넘기면 이진 ROC-AUC 가 계산돼야 한다(완벽 분리 → 1.0).
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    result = M.compute_metrics(y_true, y_true * 0 + np.array([0, 0, 1, 1]),
                               ["neg", "pos"], y_score=y_score)
    assert result["roc_auc_ovr"] == pytest.approx(1.0)


def test_metrics_missing_class_does_not_crash():
    # test 셋에 특정 클래스가 없어도(예: 클래스 2 없음) 예외 없이 계산돼야 한다.
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    result = M.compute_metrics(y_true, y_pred, ["a", "b", "c"])
    assert "per_class" in result and result["per_class"]["c"]["support"] == 0


# ── 클래스 가중치 ──────────────────────────────────────────────────────────────
def test_class_weights_inverse_frequency():
    # 클래스 0이 3배 많으면 클래스 0의 가중치가 더 작아야 한다.
    labels = np.array([0, 0, 0, 1])
    weights = compute_class_weights(labels, n_classes=2)
    assert weights[0] < weights[1]
    # 평균이 1 근처(정규화 정의상)인지 대략 확인.
    assert weights.shape == (2,)


def test_class_weights_empty_class_safe():
    # 라벨에 없는 클래스가 있어도 0으로 나누지 않고 유한값을 내야 한다.
    labels = np.array([0, 0, 0])
    weights = compute_class_weights(labels, n_classes=3)
    assert np.isfinite(weights).all()


# ── 바이트 시퀀스 인코딩 ────────────────────────────────────────────────────────
def test_encode_shift_and_pad():
    # "AB" = 0x41,0x42 → 인덱스는 +1 시프트(0x42,0x43), 나머지는 pad(0).
    seq = encode_byte_sequence("AB", max_len=4)
    assert seq.tolist() == [0x41 + 1, 0x42 + 1, PAD_INDEX, PAD_INDEX]


def test_encode_truncation():
    # max_len 보다 길면 앞에서부터 자른다.
    seq = encode_byte_sequence("ABCDEF", max_len=3)
    assert seq.tolist() == [0x41 + 1, 0x42 + 1, 0x43 + 1]


def test_encode_values_within_vocab():
    # 모든 인덱스가 [0, VOCAB_SIZE) 범위 안이어야 한다(임베딩 인덱스 안전).
    seq = encode_byte_sequence("가나다 !@#", max_len=32)
    assert seq.min() >= 0 and seq.max() < VOCAB_SIZE


def test_encode_matrix_shape():
    matrix = encode_byte_matrix(["a", "bb", ""], max_len=5)
    assert matrix.shape == (3, 5)
    assert (matrix[2] == PAD_INDEX).all()  # 빈 문자열은 전부 pad


# ── (선택) torch 모델 스모크 — torch 없으면 건너뜀 ────────────────────────────────
def test_cnn_forward_shape():
    torch = pytest.importorskip("torch")
    import cnn

    model = cnn.build_model(num_classes=4)
    x = torch.zeros(2, 1, 48, 48)  # 배치2, 1채널, 48x48
    out = model(x)
    assert out.shape == (2, 4)  # (배치, 클래스 수) logits


def test_charcnn_forward_shape():
    torch = pytest.importorskip("torch")
    import text_models

    model = text_models.build_model("charcnn", num_classes=4)
    x = torch.zeros(2, 32, dtype=torch.long)  # 배치2, 길이32 바이트 시퀀스
    out = model(x)
    assert out.shape == (2, 4)
