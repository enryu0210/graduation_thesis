"""
Phase 3-B — 페이로드(문자열) → 그레이스케일 이미지 변환 핵심 모듈

목적:
    설계 문서 4장 "페이로드 → 이미지 변환 설계"의 Step 1~2 를 구현한다.
      Step 1) 페이로드 문자열을 UTF-8 바이트 시퀀스로 인코딩
      Step 2) 고정 폭 W(=정사각 한 변)로 2D 리셰이프 (Nataraj 방식)
              - 목표 크기(side×side)보다 짧으면 zero-padding, 길면 truncation
              - 픽셀 값 = 바이트 값(0~255)

기본 W:
    EDA(docs/EDA_notes.md) 결과에 따라 48(48×48, 2,304B)을 기본값으로 둔다.
    (페이로드 98.6% 를 잘림 없이 담는 최소 정사각 크기)

설계 원칙:
    - 이 모듈은 "순수 변환 함수"만 담는다(파일 I/O·데이터셋 로딩은 build_image_dataset.py 로 분리).
      → 단위 테스트가 쉽고, 학습 파이프라인 어디서든 재사용 가능.
    - 반환 이미지는 uint8(0~255)로 저장하고, 모델 입력 직전에 normalize_01 로 0~1 스케일링한다.
      (원본 바이트 정보를 손실 없이 보관하기 위해 저장은 uint8 유지)
"""

from __future__ import annotations

import numpy as np

# EDA 로 결정한 기본 정사각 한 변(픽셀). 근거: docs/EDA_notes.md
DEFAULT_SIDE = 48


def text_to_bytes(text: str) -> bytes:
    """문자열을 UTF-8 바이트로 인코딩한다.

    errors='replace': 인코딩 불가 문자가 있어도 예외로 죽지 않고 대체 바이트로 처리
    (전처리에서 raw 를 그대로 보존하므로, 여기서 극히 드문 이상 문자는 관대하게 다룬다).
    """
    return text.encode("utf-8", errors="replace")


def bytes_to_image(data: bytes, side: int = DEFAULT_SIDE) -> np.ndarray:
    """바이트 시퀀스를 (side, side) uint8 그레이스케일 이미지로 변환한다.

    - side*side 보다 길면 앞에서부터 잘라낸다(truncation).
    - 짧으면 뒤를 0으로 채운다(zero-padding).
    - 행 우선(row-major)으로 2D 리셰이프한다.

    왜 앞에서 자르나:
        웹 공격 페이로드는 앞부분(구문 시작)에 판별 정보가 몰리는 경향이 있어
        뒤쪽을 버리는 편이 정보 손실이 적다. (이 가정은 논문에서 명시/검증)
    """
    if side <= 0:
        raise ValueError(f"side 는 양의 정수여야 합니다: {side}")

    capacity = side * side
    # 앞에서부터 capacity 만큼만 사용(길면 잘림).
    truncated = data[:capacity]

    # 고정 길이 버퍼(0으로 초기화)에 실제 바이트를 채워 넣는다 → 자동 zero-padding.
    flat = np.zeros(capacity, dtype=np.uint8)
    if truncated:
        flat[: len(truncated)] = np.frombuffer(truncated, dtype=np.uint8)

    return flat.reshape(side, side)


def payload_to_image(text: str, side: int = DEFAULT_SIDE) -> np.ndarray:
    """문자열 페이로드를 곧바로 (side, side) uint8 이미지로 변환하는 편의 함수."""
    return bytes_to_image(text_to_bytes(text), side=side)


def normalize_01(image: np.ndarray) -> np.ndarray:
    """uint8(0~255) 이미지를 float32(0~1)로 스케일링한다 (모델 입력 직전 사용)."""
    return image.astype(np.float32) / 255.0
