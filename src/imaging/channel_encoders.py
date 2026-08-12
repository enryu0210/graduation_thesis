"""
Phase 3-B(확장) — RGB 채널 인코더 모듈 (교수 피드백: "G/B 채널 값을 바꿔볼 것")

목적:
    페이로드 바이트 시퀀스를 여러 "관점(채널)"으로 동시에 표현하기 위한 순수 변환 함수 모음.
    각 인코더는 (capacity,) uint8 바이트 배열을 받아 같은 길이의 (capacity,) uint8 채널을
    돌려준다. payload_to_image.py 가 이 채널들을 R/G/B 로 쌓아 (side, side, 3) 이미지를 만든다.

왜 "교체 가능한 인코더"로 설계했나 (핵심):
    교수님 요구가 "G·B 채널에 어떤 값을 넣을지 여러 조합을 실험(ablation)하라"이다.
    따라서 채널을 하드코딩하지 않고 이름→함수 레지스트리(ENCODERS)로 두어,
    build_image_dataset.py 에서 `--rgb-encoders raw_byte,char_class,local_entropy` 처럼
    문자열만 바꾸면 채널 조합을 갈아끼울 수 있게 했다. (코드 수정 없이 조합 실험 가능)

설계 근거(관련연구):
    - raw_byte     : Nataraj et al. 바이트-플롯(그레이스케일 malware image)의 표준 표현.
    - char_class   : 소스코드/구문 정보를 채널로 넣는 계열(VulCNN 의 그래프 중심성 채널,
                     assembly-RGB 연구의 "syntactic info in green channel")에서 착안.
                     웹 공격은 특수문자(<>'"();= 등) 구문에 신호가 몰리므로 이를 밝게 강조.
    - normalized_char_class : char_class 의 인코딩 불변 판본(M2, docs/11 §4). 퍼센트·HTML
                     엔티티를 해독한 뒤의 클래스를 칠해, `%27` 과 `'` 가 같은 값이 되게 한다.
    - local_entropy: 악성코드 엔트로피-이미지 계열(Shannon local entropy). 인코딩·난독화·
                     암호화 영역이 밝아져(고엔트로피) 표현 은닉 정도를 픽셀로 드러낸다.
                     → RQ4(암호화 경계) 서사와도 자연스럽게 연결.

주의:
    - 입력 바이트 배열에는 zero-padding(값 0x00)이 섞여 있을 수 있다. 각 인코더는 padding 을
      특별 취급하지 않고 값 0 으로 자연스럽게 처리한다(패딩 영역 = 저정보 영역으로 일관 표현).
"""

from __future__ import annotations

import re
from html.entities import html5 as _HTML5_ENTITIES
from typing import Callable

import numpy as np

# ASCII 카테고리 경계(문자 클래스 채널용). 값은 "밝기 = 구문적 중요도" 직관으로 배치.
_CHAR_CLASS_TABLE: np.ndarray | None = None


def _build_char_class_table() -> np.ndarray:
    """바이트값(0~255) → 문자 클래스 밝기 로 매핑하는 룩업테이블(256,)을 만든다.

    밝기 배치 의도(웹 공격 탐지 관점):
        - 공격 구문의 핵심인 특수문자/구두점(<>'"();=/\\&%$| 등)을 가장 밝게(255) → 신호 강조.
        - 영문 소문자/대문자/숫자는 중간 밝기로 서로 구분(토큰 종류 구별).
        - 공백류는 낮게, 제어문자·패딩(0x00)·기타는 0.
    한 번 만들어 캐시한다(매 페이로드마다 재생성 방지).
    """
    table = np.zeros(256, dtype=np.uint8)

    # 공백류(space, \t, \n, \r, \v, \f): 낮은 밝기
    for c in (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C):
        table[c] = 32

    # 숫자 '0'-'9'
    table[0x30:0x3A] = 96
    # 대문자 'A'-'Z'
    table[0x41:0x5B] = 128
    # 소문자 'a'-'z'
    table[0x61:0x7B] = 160

    # 공격 구문에서 두드러지는 특수문자/구두점 → 최고 밝기(255)로 강조.
    # (SQLi/XSS/CmdI 페이로드의 판별 신호가 이 문자들에 집중되어 있음)
    special = "<>\"'`;:(){}[]=/\\&%$|!?#@*+-.,~^"
    for ch in special:
        table[ord(ch)] = 255

    return table


def raw_byte(byte_values: np.ndarray) -> np.ndarray:
    """R 채널 기본값 — 원본 바이트 값을 그대로 사용(Nataraj 바이트-플롯)."""
    return byte_values.astype(np.uint8, copy=True)


def char_class(byte_values: np.ndarray) -> np.ndarray:
    """문자 클래스 채널 — 바이트를 ASCII 카테고리 밝기로 치환(구문 구조 강조)."""
    global _CHAR_CLASS_TABLE
    if _CHAR_CLASS_TABLE is None:
        _CHAR_CLASS_TABLE = _build_char_class_table()
    return _CHAR_CLASS_TABLE[byte_values]


# ---------------------------------------------------------------------------
# M2(Phase 12) — 인코딩 불변 문자 클래스 채널
# ---------------------------------------------------------------------------
# 왜 만들었나(docs/11 §4):
#     RQ3 실측에서 적대적 증강(학습형 방어)이 **미지의 인코딩 변형에 무력**했다(F9).
#     원인은 회피 위력의 대부분이 url_encode·double_url_encode 같은 "인코딩 계열"인데
#     그 계열을 학습에서 빼면(S-A) 배울 것 자체가 없기 때문이다.
#     → 학습으로 못 얻는 불변성은 **표현(채널)에 구조로 넣는다.**
#
# ⚠️ R 채널(raw_byte)은 그대로 둔다는 것이 정규화 방어(docs/10 §3 B안)와의 결정적 차이다.
#     B안은 디코딩한 결과만 보므로 원본 정보를 잃지만(clean MCC −1.04pp 지불),
#     채널로 넣으면 원본 뷰와 정규화 뷰를 동시에 본다 → "인코딩을 썼다는 사실"도 신호로 남는다.

# 디코딩 반복 상한. preprocess.MAX_DECODE_ROUNDS(=3)와 같은 값으로 맞춘다
# (B안과 같은 조건에서 비교해야 M2 의 이득이 "더 깊이 디코딩해서"가 아님이 분명해진다).
_MAX_DECODE_ROUNDS = 3

# `%XX` 와 다중 인코딩 `%2527`(=`%27`→`'`)·`%252527` 을 한 번에 잡는다.
# `(?:25)*` 는 "퍼센트 자신이 %25 로 다시 인코딩된 층"을 뜻한다. 그리디 매칭이라
# `%2525`(=`%`) 처럼 마지막 두 자리가 25 인 경우도 역추적으로 올바르게 처리된다.
_PERCENT_RE = re.compile(r"%(?:25)*[0-9A-Fa-f]{2}")

# HTML 엔티티 3종: 10진(`&#60;`) · 16진(`&#x3C;`) · 이름(`&lt;`).
_ENTITY_RE = re.compile(
    r"&(?:#(\d{1,7})|#[xX]([0-9A-Fa-f]{1,6})|([A-Za-z][A-Za-z0-9]{1,31}));"
)

_PERCENT_BYTE = 0x25  # '%'
_AMPERSAND_BYTE = 0x26  # '&'


def _percent_value(match: re.Match) -> int:
    """`%..XX` 매치 → 최종 디코딩 바이트값. 마지막 두 자리가 실제 값이다."""
    return int(match.group(0)[-2:], 16)


def _entity_value(match: re.Match) -> int | None:
    """HTML 엔티티 매치 → 코드포인트(모르는 이름이면 None → 디코딩하지 않음)."""
    decimal, hexadecimal, name = match.group(1), match.group(2), match.group(3)
    if decimal is not None:
        code = int(decimal)
    elif hexadecimal is not None:
        code = int(hexadecimal, 16)
    else:
        # html5 테이블은 세미콜론까지 포함한 키를 쓴다(`lt;`). 없는 이름은 원문 유지.
        decoded = _HTML5_ENTITIES.get(name + ";")
        if decoded is None or len(decoded) != 1:
            return None
        code = ord(decoded)
    # 유효 코드포인트 범위를 벗어나면 디코딩하지 않는다(chr() 예외 방지).
    return code if 0 <= code <= 0x10FFFF else None


def _decode_spans(text: str) -> tuple[list[int], list[int], list[int]]:
    """인코딩을 풀되 **원본 바이트 구간을 함께 추적**한다.

    반환: (codes, starts, stops) — 디코딩 결과 문자 하나하나에 대해
          그것이 원본에서 차지하던 구간 [start, stop) 를 짝지어 준다.

    왜 구간 추적이 필요한가(핵심):
        채널 인코더는 입력 바이트 배열과 **같은 길이**를 돌려줘야 한다(R/G/B 정렬).
        그래서 "디코딩한 문자열"을 그냥 만들면 안 되고, `%27`(3바이트)이 `'`로 풀렸다는
        사실을 **그 3칸 전부에 같은 값**을 칠하는 방식으로 표현해야 한다.
        → 길이는 유지되고, 값은 인코딩 여부와 무관해진다(= 인코딩 불변).

    ⚠️ `+`→공백 치환은 하지 않는다. urllib 의 unquote_plus 는 그렇게 하지만,
       `+` 는 SQLi/XSS 페이로드에서 실제 연산자로 쓰이는 구문 신호다. 여기서 지우면
       "인코딩 불변"을 얻는 대신 진짜 신호를 잃는다.
    """
    codes = [ord(ch) for ch in text]
    starts = list(range(len(text)))
    stops = list(range(1, len(text) + 1))

    for _ in range(_MAX_DECODE_ROUNDS):
        changed = False
        # 퍼센트 → 엔티티 순서로 한 라운드. 라운드를 반복하는 이유는 **계열이 섞인 중첩**
        # 때문이다(예: `&lt;` 에 URL 인코딩을 덧씌운 `%26lt%3B`. 퍼센트를 먼저 풀어야
        # `&lt;` 가 드러나고, 그 다음 라운드에서 엔티티로 풀린다).
        for pattern, decode in ((_PERCENT_RE, _percent_value), (_ENTITY_RE, _entity_value)):
            current = "".join(map(chr, codes))
            new_codes: list[int] = []
            new_starts: list[int] = []
            new_stops: list[int] = []
            pos = 0
            for match in pattern.finditer(current):
                value = decode(match)
                if value is None:  # 해독 불가 → 원문 그대로 둔다
                    continue
                i, j = match.span()
                new_codes.extend(codes[pos:i])
                new_starts.extend(starts[pos:i])
                new_stops.extend(stops[pos:i])
                # 매치 전체가 문자 하나로 접힌다. 구간은 원본 좌표에서 [첫 칸, 마지막 칸).
                new_codes.append(value)
                new_starts.append(starts[i])
                new_stops.append(stops[j - 1])
                pos = j
                changed = True
            if pos:  # 이번 패스에서 실제로 접힌 게 있을 때만 교체
                new_codes.extend(codes[pos:])
                new_starts.extend(starts[pos:])
                new_stops.extend(stops[pos:])
                codes, starts, stops = new_codes, new_starts, new_stops
        if not changed:
            break

    return codes, starts, stops


def normalized_char_class(byte_values: np.ndarray) -> np.ndarray:
    """인코딩 불변 문자 클래스 채널 — `%XX`·`&#NN`·`&name;` 을 **해독한 뒤의** 클래스를 칠한다.

    예시(왜 불변인가):
        `'`      → 1바이트, 값 255(특수문자)
        `%27`    → 3바이트, **세 칸 모두 255**  ← 같은 값
        `%2527`  → 5바이트, 다섯 칸 모두 255    ← 이중 인코딩도 같은 값
        `&lt;`   → 4바이트, 네 칸 모두 255(`<` 의 클래스)
        인코딩을 쓰면 픽셀 **개수**는 늘지만 **값**은 원문과 같아진다. 그래서 학습이
        인코딩 변종을 본 적 없어도 같은 패턴으로 보인다(규칙 기반이라 원리적 일반화).

    구현 메모:
        - 인코딩이 없으면 char_class 와 완전히 동일하다 → clean 성능 손실이 구조적으로 작다.
        - 빠른 경로: 버퍼에 '%' 도 '&' 도 없으면 룩업 한 번으로 끝낸다(대부분의 정상 트래픽).
        - zero-padding 구간은 디코딩 대상에서 제외한다(불필요한 파이썬 스캔 회피).
    """
    global _CHAR_CLASS_TABLE
    if _CHAR_CLASS_TABLE is None:
        _CHAR_CLASS_TABLE = _build_char_class_table()

    out = _CHAR_CLASS_TABLE[byte_values]  # 기준값 = 원본 바이트의 문자 클래스

    # 인코딩 표식이 아예 없으면 char_class 와 같다 → 파이썬 스캔 생략.
    has_marker = np.any(byte_values == _PERCENT_BYTE) or np.any(byte_values == _AMPERSAND_BYTE)
    if not has_marker:
        return out

    # 뒤쪽 zero-padding 을 잘라낸 실제 페이로드 구간만 본다.
    nonzero = np.nonzero(byte_values)[0]
    if nonzero.size == 0:
        return out
    end = int(nonzero[-1]) + 1

    # 바이트↔문자가 1:1 무손실인 latin-1 로 문자열화(UTF-8 다중바이트는 그대로 통과).
    text = byte_values[:end].tobytes().decode("latin-1")
    codes, starts, stops = _decode_spans(text)

    out = out.copy()  # 룩업 결과는 새 배열이지만, 의도를 드러내기 위해 명시적으로 복사
    for code, start, stop in zip(codes, starts, stops):
        if stop - start == 1:
            continue  # 접히지 않은 칸 = 원본 그대로 → 기준값이 이미 정답
        # 디코딩된 문자의 클래스를, 그 문자가 원본에서 차지하던 칸 전부에 칠한다.
        value = int(_CHAR_CLASS_TABLE[code]) if code < 256 else 0
        out[start:stop] = value

    return out


def local_entropy(byte_values: np.ndarray, window: int = 8) -> np.ndarray:
    """지역 엔트로피 채널 — 각 위치 주변 window 바이트의 Shannon 엔트로피(0~255 스케일).

    의미:
        평문(구조적, 저엔트로피)은 어둡고, Base64/암호화(고엔트로피)는 밝아진다.
        → "표현이 얼마나 내용을 감췄는가"를 픽셀 밝기로 시각화(RQ4 서사와 연결).

    구현(벡터화):
        - 앞뒤로 window//2 만큼 0-패딩한 뒤 sliding window(모든 위치에 정렬)로 (L, window) 를 만든다.
        - 각 창의 엔트로피를 창 내 실제 카운트로 계산한다. 파이썬 루프 없이 정렬+그룹핑으로 벡터화.
        - 최대 엔트로피 log2(window) 로 나눠 0~255 로 스케일.
    """
    if window < 2:
        raise ValueError(f"window 는 2 이상이어야 합니다: {window}")

    n = byte_values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    half = window // 2
    padded = np.pad(byte_values, (half, window - half - 1), mode="constant", constant_values=0)
    # (n, window): 각 원본 위치 i 에 대응하는 주변 창
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)

    # 각 창을 정렬해 같은 값의 런(run)을 인접시킨다.
    s = np.sort(windows, axis=1)
    # 이전 원소와 다른 지점 = 새 그룹 시작
    is_new_group = np.ones_like(s, dtype=bool)
    is_new_group[:, 1:] = s[:, 1:] != s[:, :-1]
    # 각 원소의 그룹 인덱스(행 내부 0-based). 그룹 수는 최대 window.
    group_idx = np.cumsum(is_new_group, axis=1) - 1

    # 행별로 겹치지 않는 전역 bin id 를 만들어 한 번의 bincount 로 그룹 크기를 구한다.
    row_offsets = (np.arange(n) * window)[:, None]
    global_bins = group_idx + row_offsets
    group_sizes_flat = np.bincount(global_bins.ravel(), minlength=n * window).reshape(n, window)
    # 각 원소가 속한 그룹의 크기(=해당 바이트값의 창 내 빈도)
    count_per_elem = np.take_along_axis(group_sizes_flat, group_idx, axis=1)

    # 엔트로피 = -(1/w) * Σ log2(p),  p = count/w   (원소별 기여를 합산한 수학적 동치식)
    p = count_per_elem / window
    entropy = -(1.0 / window) * np.sum(np.log2(p), axis=1)  # (n,), 단위: bit

    scaled = entropy / np.log2(window) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def bit_popcount(byte_values: np.ndarray) -> np.ndarray:
    """(ablation 후보) 바이트의 1-비트 개수(0~8) → 0~255 스케일. 바이트 밀도의 다른 관점."""
    popcount = np.unpackbits(byte_values.astype(np.uint8)[:, None], axis=1).sum(axis=1)
    return (popcount * (255 // 8)).astype(np.uint8)


def structural_special(byte_values: np.ndarray) -> np.ndarray:
    """(ablation 후보) 공격 특수문자만 255, 나머지 0 인 이진 강조 채널(구문 위치 마스크)."""
    global _CHAR_CLASS_TABLE
    if _CHAR_CLASS_TABLE is None:
        _CHAR_CLASS_TABLE = _build_char_class_table()
    # char_class 에서 255(특수문자)로 매핑된 바이트만 남긴다.
    return np.where(_CHAR_CLASS_TABLE[byte_values] == 255, 255, 0).astype(np.uint8)


def byte_delta(byte_values: np.ndarray) -> np.ndarray:
    """(ablation 후보) 직전 바이트와의 절대 변화량. 반복/구조 패턴은 어둡고 급변부는 밝다."""
    delta = np.zeros_like(byte_values, dtype=np.int16)
    delta[1:] = np.abs(byte_values[1:].astype(np.int16) - byte_values[:-1].astype(np.int16))
    return np.clip(delta, 0, 255).astype(np.uint8)


# 이름 → 인코더 함수 레지스트리.
# build_image_dataset.py 의 --rgb-encoders 문자열이 이 키를 가리킨다.
# 교수님 요구(G/B 채널 조합 실험)를 코드 수정 없이 하려면 여기에만 추가하면 된다.
ENCODERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw_byte": raw_byte,
    "char_class": char_class,
    "normalized_char_class": normalized_char_class,
    "local_entropy": local_entropy,
    "bit_popcount": bit_popcount,
    "structural_special": structural_special,
    "byte_delta": byte_delta,
}

# RGB 기본 채널 조합(R, G, B). 설계문서 §4 Step 3 + 관련연구 근거.
DEFAULT_RGB_ENCODERS: tuple[str, str, str] = ("raw_byte", "char_class", "local_entropy")


def get_encoder(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """이름으로 인코더 함수를 찾는다(없으면 명확한 에러)."""
    if name not in ENCODERS:
        available = ", ".join(sorted(ENCODERS))
        raise KeyError(f"알 수 없는 채널 인코더 '{name}'. 사용 가능: {available}")
    return ENCODERS[name]
