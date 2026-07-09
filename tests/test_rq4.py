"""RQ4a 표현 은닉 변환(encoding.py) 단위 테스트.

검증 관점(설계 docs/06 §1):
    - 스윕이 실제로 '은닉 강도'를 키우는가(엔트로피 단조 증가 경향).
    - 랜덤 암호화(S4)가 진짜 랜덤 성질을 갖는가(고엔트로피 + 같은 평문→다른 암호문).
    - 가역 인코딩(S1/S2)이 무손실인가(왕복 복원).
    - 바이트→문자열 변환이 무손실인가(TF-IDF 입력 안전성).
"""

import base64
from urllib.parse import unquote_to_bytes

import encoding as E

# 대표 공격 페이로드 샘플(각 클래스 성격)
SAMPLES = [
    "1' OR 1=1 -- ",
    "<script>alert(1)</script>",
    "; cat /etc/passwd",
    "admin' UNION SELECT username, password FROM users--",
]


def test_all_stages_return_bytes():
    """모든 단계가 bytes 를 돌려주고 빈 결과가 아니어야 한다."""
    for _code, _name, fn, _rev in E.STAGES:
        for s in SAMPLES:
            out = fn(s)
            assert isinstance(out, bytes) and len(out) > 0


def test_entropy_increases_toward_encryption():
    """평문(S0) 대비 AES(S4)의 엔트로피가 뚜렷이 높아야 한다(은닉 강도 증가)."""
    for s in SAMPLES:
        e_raw = E.byte_entropy(E.s0_raw(s))
        e_aes = E.byte_entropy(E.s4_aes_randomiv(s))
        assert e_aes > e_raw, f"AES 엔트로피가 평문보다 낮음: {s!r}"


def test_aes_is_randomized():
    """S4 는 랜덤 IV 라 '같은 평문 → 매번 다른 암호문'이어야 한다(재학습 무력화의 근거)."""
    s = SAMPLES[0]
    assert E.s4_aes_randomiv(s) != E.s4_aes_randomiv(s)


def test_aes_high_entropy():
    """충분히 긴 페이로드의 AES 출력은 고엔트로피(≈난수)여야 한다."""
    long_payload = SAMPLES[3] * 20  # 길이를 키워 엔트로피 추정을 안정화
    assert E.byte_entropy(E.s4_aes_randomiv(long_payload)) > 7.0


def test_xor_is_deterministic_and_reversible():
    """S3 고정키 XOR 은 결정론적(같은 입력→같은 출력)이고 왕복 복원돼야 한다."""
    s = SAMPLES[0]
    out1 = E.s3_xor_fixedkey(s)
    out2 = E.s3_xor_fixedkey(s)
    assert out1 == out2  # 결정론적
    # XOR 을 한 번 더 적용하면 원문 바이트로 복원(가역성)
    key = E._XOR_KEY
    restored = bytes(b ^ key[i % len(key)] for i, b in enumerate(out1))
    assert restored == s.encode("utf-8")


def test_keyless_encodings_are_lossless():
    """S1(URL)·S2(Base64)는 키 없는 가역 인코딩 → 원문 복원 가능."""
    for s in SAMPLES:
        assert unquote_to_bytes(E.s1_urlencode(s)) == s.encode("utf-8")
        assert base64.b64decode(E.s2_base64(s)) == s.encode("utf-8")


def test_latin1_roundtrip_is_lossless():
    """임의 바이트(암호문 포함)를 latin-1 로 문자열화해도 무손실이어야 한다(TF-IDF 입력)."""
    data = E.s4_aes_randomiv(SAMPLES[0])
    assert E.bytes_to_latin1(data).encode("latin-1") == data
