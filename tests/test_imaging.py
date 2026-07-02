"""바이트→이미지 변환 함수(payload_to_image.py)의 단위 테스트.

검증 포인트:
    - 출력 shape/dtype 이 항상 (side, side) uint8 인가
    - 짧은 입력은 zero-padding, 긴 입력은 truncation 이 정확한가
    - 바이트 값이 픽셀 값으로 손실 없이 매핑되는가
    - 멀티바이트(UTF-8) 문자·빈 문자열 등 경계 상황을 안전하게 처리하는가
"""

import numpy as np
import pytest

from payload_to_image import (
    DEFAULT_SIDE,
    bytes_to_image,
    normalize_01,
    payload_to_image,
    text_to_bytes,
)


def test_shape_and_dtype():
    img = payload_to_image("SELECT * FROM users", side=16)
    assert img.shape == (16, 16)
    assert img.dtype == np.uint8


def test_default_side():
    img = payload_to_image("test")
    assert img.shape == (DEFAULT_SIDE, DEFAULT_SIDE)


def test_zero_padding_short_input():
    # "AB" = 0x41,0x42 → 첫 두 픽셀만 값이 있고 나머지는 0 이어야 한다.
    img = bytes_to_image(b"AB", side=4)
    flat = img.flatten()
    assert flat[0] == 0x41
    assert flat[1] == 0x42
    assert (flat[2:] == 0).all()  # 나머지는 zero-padding


def test_truncation_long_input():
    # side=2 → 용량 4바이트. 6바이트를 넣으면 앞 4개만 남아야 한다.
    img = bytes_to_image(b"ABCDEF", side=2)
    assert img.flatten().tolist() == [0x41, 0x42, 0x43, 0x44]


def test_row_major_reshape():
    # 4바이트를 2x2 로 넣으면 행 우선으로 배치되어야 한다.
    img = bytes_to_image(bytes([1, 2, 3, 4]), side=2)
    assert img.tolist() == [[1, 2], [3, 4]]


def test_empty_string_all_zeros():
    img = payload_to_image("", side=8)
    assert img.shape == (8, 8)
    assert (img == 0).all()


def test_multibyte_utf8():
    # 한글 한 글자는 UTF-8 에서 3바이트 → 값이 채워지고 예외 없이 처리돼야 한다.
    raw = text_to_bytes("가")
    assert len(raw) == 3
    img = payload_to_image("가", side=4)
    assert img.flatten()[:3].tolist() == list(raw)


def test_value_range_preserved():
    # 0~255 전 범위 바이트가 그대로 픽셀 값으로 보존되는지 확인.
    data = bytes(range(256))
    img = bytes_to_image(data, side=16)  # 정확히 256바이트 = 16x16
    assert img.min() == 0 and img.max() == 255
    assert sorted(img.flatten().tolist()) == list(range(256))


def test_normalize_01_range():
    img = bytes_to_image(bytes([0, 128, 255, 0]), side=2)
    norm = normalize_01(img)
    assert norm.dtype == np.float32
    assert norm.min() == 0.0 and norm.max() == 1.0
    assert np.isclose(norm.flatten()[1], 128 / 255)


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        bytes_to_image(b"x", side=0)
