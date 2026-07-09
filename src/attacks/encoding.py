"""
Phase 6 (RQ4a) — 표현 은닉 변환: 평문 → 인코딩 → 암호화 스윕

목적:
    페이로드가 "얼마나 가려졌는가"를 단계적으로 재현한다. 각 단계는 페이로드 문자열을
    받아 **네트워크상에서 관측될 바이트열**을 돌려주는 순수 함수다(부작용 없음).
    이 바이트열로 (1) 엔트로피, (2) 이미지 클래스 분리도, (3) 재학습한 탐지기의 성능을
    측정해 "내용 기반 탐지가 어디서 붕괴하는가"를 본다(docs/06 §1 RQ4a).

핵심 설계 판단 — 왜 '가역 인코딩'과 '랜덤 암호화'를 구분하는가:
    - Base64/URL 인코딩은 **키 없는 가역 변환**이다. 개별 메시지 엔트로피는 올라가도
      "같은 평문 → 같은 출력"이라 클래스 구조가 보존된다. 따라서 탐지기를 그 표현으로
      **재학습하면 성능이 되살아난다**. → "인코딩은 (재학습 앞에서) 탐지를 못 막는다".
    - 랜덤 IV 암호화(AES-CTR, 매 메시지 새 nonce)는 **같은 평문이 매번 다른 암호문**이 된다.
      클래스 조건부 구조가 사라져 **재학습해도** 배울 게 없다. → 이것이 진짜 탐지 경계.
    이 대비가 RQ4a 의 논지다. 그래서 단계에 둘 다 넣는다.

바이트를 문자열 모델(TF-IDF)에 넣는 법:
    암호문/Base64 는 유효 UTF-8 이 아닐 수 있다. char n-gram TF-IDF 에 먹이려면
    바이트↔문자가 1:1 무손실인 latin-1 로 디코딩해 문자열화한다(bytes_to_latin1).
"""

from __future__ import annotations

import base64
import math
import os
from collections import Counter
from urllib.parse import quote

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# ── 고정 키(재현성용). nonce 는 매 호출 랜덤(= 랜덤 암호화의 본질). ──────────────
# 논문 관점: 키 값 자체는 중요치 않다. 중요한 건 "매 메시지 nonce 가 달라
# 같은 평문이 다른 암호문이 된다"는 성질이며, 그게 재학습을 무력화한다.
_AES_KEY = bytes.fromhex("00112233445566778899aabbccddeeff")  # 16B = AES-128
_XOR_KEY = b"\x5a\x3c\x71"  # 고정 반복키(결정론적 XOR — 가역·구조보존 단계)


def s0_raw(text: str) -> bytes:
    """S0 — 평문. UTF-8 바이트 그대로(RQ1 의 text_raw 와 동일 관측)."""
    return text.encode("utf-8", errors="replace")


def s1_urlencode(text: str) -> bytes:
    """S1 — URL 퍼센트 인코딩(키 없는 가역). 특수문자를 %XX 로 확장."""
    return quote(text, safe="").encode("ascii", errors="replace")


def s2_base64(text: str) -> bytes:
    """S2 — Base64(키 없는 가역). 문자셋 축소·재배치로 중엔트로피."""
    return base64.b64encode(text.encode("utf-8", errors="replace"))


def s3_xor_fixedkey(text: str) -> bytes:
    """S3 — 고정 반복키 XOR(결정론적·가역). 준랜덤이지만 '같은 평문 → 같은 출력'.

    → 개별 엔트로피는 높아 보여도 클래스 구조가 보존되므로, 재학습한 탐지기는
      여전히 상당 부분 배울 수 있다(가역 변환의 특성). 이 점을 실측으로 보인다.
    """
    data = text.encode("utf-8", errors="replace")
    key = _XOR_KEY
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def s4_aes_randomiv(text: str) -> bytes:
    """S4 — AES-CTR, 매 호출 랜덤 nonce(= 랜덤 암호화). 같은 평문도 매번 다른 암호문.

    반환: nonce(16B) + ciphertext. 둘 다 고엔트로피라 이미지/엔트로피 관점에서 난수.
    → 클래스 조건부 구조가 사라져 **재학습해도** 탐지가 불가능해진다(진짜 경계).
    """
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.AES(_AES_KEY), modes.CTR(nonce))
    enc = cipher.encryptor()
    ct = enc.update(text.encode("utf-8", errors="replace")) + enc.finalize()
    return nonce + ct


# 스윕 순서(엔트로피/은닉 강도 증가 방향). (코드, 사람이 읽는 이름, 함수, 가역성 메모)
STAGES = [
    ("S0_raw", "평문", s0_raw, "keyless"),
    ("S1_urlencode", "URL인코딩", s1_urlencode, "keyless"),
    ("S2_base64", "Base64", s2_base64, "keyless"),
    ("S3_xor", "고정키XOR", s3_xor_fixedkey, "deterministic"),
    ("S4_aes", "AES랜덤IV", s4_aes_randomiv, "randomized"),
]


def byte_entropy(data: bytes) -> float:
    """바이트열의 Shannon 엔트로피(bits/byte, 0~8). 8 에 가까울수록 난수적."""
    if not data:
        return 0.0
    n = len(data)
    counts = Counter(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def bytes_to_latin1(data: bytes) -> str:
    """바이트열을 무손실 문자열로(각 바이트 → 코드포인트). TF-IDF 입력용."""
    return data.decode("latin-1")
