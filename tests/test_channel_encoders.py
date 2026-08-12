"""M2(Phase 12) — 인코딩 불변 채널 인코더 `normalized_char_class` 의 단위 테스트.

이 채널의 존재 이유가 단 한 줄로 요약되기 때문에 테스트도 그 한 줄을 중심으로 짰다:
    "같은 페이로드라면, 인코딩을 씌우든 안 씌우든 **채널 값이 같아야 한다.**"

검증 포인트:
    - 왕복 불변성: `'` 와 `%27` 과 `%2527` 과 `&#39;` 가 전부 같은 채널 값을 낸다
    - 길이 보존: 인코더 계약(입력과 같은 길이의 uint8) 을 어기지 않는다
    - 하위호환: 인코딩이 없는 입력에서는 기존 char_class 와 완전히 동일하다
    - 안전성: 깨진 인코딩·모르는 엔티티·빈 입력에서 죽지 않고 원문 취급한다
    - 레지스트리/약어: 파일명 약어가 저장·로드 두 곳에서 일치한다(덮어쓰기 사고 방지)
"""

import numpy as np
import pytest

from channel_encoders import (
    ENCODERS,
    char_class,
    get_encoder,
    normalized_char_class,
)


def _channel(text: str, capacity: int = 64) -> np.ndarray:
    """문자열을 실제 파이프라인과 같은 방식(UTF-8 → zero-padded 버퍼)으로 채널화한다."""
    data = text.encode("utf-8")
    buffer = np.zeros(capacity, dtype=np.uint8)
    buffer[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    return normalized_char_class(buffer)


def _values_of(text: str, length: int) -> np.ndarray:
    """페이로드가 실제로 차지한 앞부분 length 칸의 채널 값."""
    return _channel(text)[:length]


# ---------------------------------------------------------------------------
# 핵심 — 인코딩 불변성
# ---------------------------------------------------------------------------

def test_percent_encoding_matches_plain_character():
    """`%27`(3바이트)의 세 칸이 전부 `'` 의 클래스와 같아야 한다."""
    plain = _values_of("'", 1)
    encoded = _values_of("%27", 3)
    assert (encoded == plain[0]).all()


def test_double_percent_encoding_matches_plain_character():
    """이중 인코딩 `%2527` 도 마찬가지 — 학습이 본 적 없어도 같은 값으로 보인다."""
    plain = _values_of("'", 1)
    encoded = _values_of("%2527", 5)
    assert (encoded == plain[0]).all()


def test_html_named_entity_matches_plain_character():
    plain = _values_of("<", 1)
    encoded = _values_of("&lt;", 4)
    assert (encoded == plain[0]).all()


def test_html_decimal_and_hex_entities_match_plain_character():
    plain = _values_of("<", 1)[0]
    assert (_values_of("&#60;", 5) == plain).all()
    assert (_values_of("&#x3C;", 6) == plain).all()


def test_cross_scheme_nesting_url_over_entity():
    """`&lt;` 위에 URL 인코딩을 덧씌운 `%26lt%3B` — 계열이 섞인 중첩도 풀려야 한다.

    (조합 공격은 변형을 최대 5개까지 쌓으므로 이 경우가 실제로 발생한다.)
    """
    plain = _values_of("<", 1)[0]
    nested = _values_of("%26lt%3B", 8)
    assert (nested == plain).all()


def test_full_payload_invariance():
    """실제 XSS 페이로드 전체에서, 인코딩본의 채널 값 '집합'이 원문과 같아야 한다.

    길이는 달라지므로(픽셀 개수는 늘어난다) 값의 시퀀스를 런-렝스로 압축해 비교한다.
    """
    payload = "<script>alert('1')</script>"

    def compressed(text: str) -> list[int]:
        values = _channel(text, capacity=256)[: len(text.encode())]
        out: list[int] = []
        for v in values.tolist():
            if not out or out[-1] != v:
                out.append(v)
        return out

    from urllib.parse import quote
    assert compressed(quote(payload, safe="")) == compressed(payload)


# ---------------------------------------------------------------------------
# 계약 — 길이·dtype·하위호환
# ---------------------------------------------------------------------------

def test_output_length_and_dtype_preserved():
    buffer = np.frombuffer("SELECT %27 FROM t".encode(), dtype=np.uint8)
    out = normalized_char_class(buffer)
    assert out.shape == buffer.shape
    assert out.dtype == np.uint8


def test_identical_to_char_class_without_encoding():
    """인코딩 표식이 없으면 기존 char_class 와 완전히 같다(clean 성능 손실의 구조적 상한)."""
    buffer = np.frombuffer("SELECT * FROM users WHERE id=1 OR 1=1".encode(), dtype=np.uint8)
    assert np.array_equal(normalized_char_class(buffer), char_class(buffer))


def test_non_encoding_ampersand_left_alone():
    """`&` 가 쿼리 구분자로 쓰인 정상 트래픽은 건드리지 않는다(오탐 위험 차단)."""
    buffer = np.frombuffer("a=1&b=2&c=3".encode(), dtype=np.uint8)
    assert np.array_equal(normalized_char_class(buffer), char_class(buffer))


def test_padding_region_stays_zero():
    out = _channel("%27", capacity=64)
    assert (out[3:] == 0).all()


# ---------------------------------------------------------------------------
# 안전성 — 깨진 입력에서 죽지 않는다
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["%", "%2", "%ZZ", "&", "&;", "&notarealentity;", "&#;", "&#999999999;"])
def test_malformed_encodings_are_left_as_literals(text: str):
    buffer = np.zeros(32, dtype=np.uint8)
    data = text.encode()
    buffer[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    # 해독 불가는 원문 취급 → char_class 와 같아야 하고, 예외가 나면 안 된다.
    assert np.array_equal(normalized_char_class(buffer), char_class(buffer))


def test_empty_buffer():
    assert normalized_char_class(np.zeros(0, dtype=np.uint8)).shape == (0,)


def test_all_padding_buffer():
    out = normalized_char_class(np.zeros(16, dtype=np.uint8))
    assert (out == 0).all()


def test_multibyte_utf8_passes_through():
    """한글 등 다중바이트는 디코딩 대상이 아니며 그대로 통과해야 한다."""
    buffer = np.frombuffer("한글%27".encode("utf-8"), dtype=np.uint8)
    out = normalized_char_class(buffer)
    assert out.shape == buffer.shape
    # 마지막 3칸(%27)만 특수문자 클래스로 접힌다.
    assert (out[-3:] == char_class(np.frombuffer(b"'", dtype=np.uint8))[0]).all()


# ---------------------------------------------------------------------------
# 레지스트리 / 파일명 약어 — 저장·로드 불일치는 산출물 덮어쓰기로 이어진다
# ---------------------------------------------------------------------------

def test_registered_in_encoder_registry():
    assert ENCODERS["normalized_char_class"] is normalized_char_class
    assert get_encoder("normalized_char_class") is normalized_char_class


def test_encoder_abbreviation_tables_agree():
    """`_ENCODER_ABBR` 는 저장(build_image_dataset)·로드(data_image) 2곳에 중복돼 있다.

    어긋나면 학습이 엉뚱한 npz 를 찾거나 다른 조합 결과를 덮어쓴다(CLAUDE.md 경고).
    """
    from build_image_dataset import _ENCODER_ABBR as save_side
    from data_image import _ENCODER_ABBR as load_side
    assert save_side == load_side
    assert save_side["normalized_char_class"] == "nc"


def test_channel_suffix_for_m2_combination():
    from build_image_dataset import dataset_suffix
    from data_image import _channel_suffix

    encoders = ("raw_byte", "normalized_char_class", "local_entropy")
    assert dataset_suffix("rgb", encoders) == "_rgb-rb-nc-le"
    assert _channel_suffix("rgb", encoders) == "_rgb-rb-nc-le"
