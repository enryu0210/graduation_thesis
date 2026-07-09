"""
Phase 5 (RQ2) — 의미보존 회피 변형 규칙 (WAF 우회 시뮬레이션)

목적:
    설계 docs/05 §4 카탈로그를 구현한다. 각 함수는 공격 페이로드 문자열을 받아
    "여전히 동작하는(의미가 보존되는) 변형 페이로드"를 돌려주는 **순수 문자열 함수**다.

철칙(docs/05 §4):
    변형 후에도 페이로드가 "동작하는 공격"이어야 한다. 라이브 타깃에서 실행 검증을
    할 수 없으므로, **규칙 수준에서 의미보존이 보장되는 변형만** 쓴다(임의 문자 삽입 등
    파괴적 변형 금지). 이 제약은 논문 한계로 명시한다.

설계:
    - 결정론 옵션: 대소문자 랜덤화 등은 seed 를 받아 재현 가능하게 한다.
    - 등록제(REGISTRY): 이름→함수 매핑으로, 상위 오케스트레이터(problem_space.py)가
      단일/조합 적용을 이름으로 다루게 한다. 클래스별 적용 가능 변형도 함께 관리한다.
"""

from __future__ import annotations

import random
import re
from urllib.parse import quote

# ─────────────────────────────────────────────────────────────────────────────
# 4.1 공통(문법 무관) 변형 — 어느 공격 클래스에도 적용 가능
# ─────────────────────────────────────────────────────────────────────────────

def url_encode(payload: str, _rng: random.Random | None = None) -> str:
    """URL 퍼센트 인코딩. 서버단 디코딩으로 원 페이로드 환원(의미보존)."""
    return quote(payload, safe="")


def double_url_encode(payload: str, _rng: random.Random | None = None) -> str:
    """이중 URL 인코딩(`'`→`%27`→`%2527`). 다단 디코딩 경로 우회."""
    return quote(quote(payload, safe=""), safe="")


def random_case(payload: str, rng: random.Random | None = None) -> str:
    """알파벳의 대소문자를 랜덤화. SQL 키워드·HTML 태그는 대소문자 무관(의미보존).

    문자열 리터럴 안('...'/"...")은 값이 바뀔 수 있어 건드리지 않는다(보수적).
    가정: 따옴표가 균형 잡혀 있다고 본다. 주입 페이로드처럼 따옴표가 불균형이면
    리터럴 안/밖 경계가 모호해질 수 있다(이 한계는 논문에 명시). 그럼에도 대부분의
    SQL 은 기본 콜레이션이 대소문자 무관이라 실무적 의미 훼손 위험은 낮다.
    """
    rng = rng or random.Random(0)
    out, in_quote, quote_ch = [], False, ""
    for ch in payload:
        if ch in ("'", '"'):
            if in_quote and ch == quote_ch:
                in_quote = False
            elif not in_quote:
                in_quote, quote_ch = True, ch
            out.append(ch)
        elif ch.isalpha() and not in_quote:
            out.append(ch.upper() if rng.random() < 0.5 else ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def space_to_comment(payload: str, _rng: random.Random | None = None) -> str:
    """공백을 인라인 주석 `/**/` 으로 치환(SQL 파서는 동일 토큰 경계로 인식)."""
    return payload.replace(" ", "/**/")


def space_to_tab(payload: str, _rng: random.Random | None = None) -> str:
    """공백을 탭으로 치환(다수 파서가 공백류로 동일 취급)."""
    return payload.replace(" ", "\t")


# ─────────────────────────────────────────────────────────────────────────────
# 4.2 SQLi 전용
# ─────────────────────────────────────────────────────────────────────────────

def sqli_inline_comment(payload: str, _rng: random.Random | None = None) -> str:
    """SQL 키워드 사이에 인라인 주석 삽입: `UNION SELECT`→`UNION/**/SELECT`."""
    return re.sub(r"\b(UNION|SELECT|FROM|WHERE|OR|AND)\s+", r"\1/**/",
                  payload, flags=re.IGNORECASE)


def sqli_logical_equiv(payload: str, _rng: random.Random | None = None) -> str:
    """논리적 동치 치환: `OR 1=1`→`OR 2=2-1`(참 값 유지, 표면형 변형)."""
    return re.sub(r"\bOR\s+1\s*=\s*1\b", "OR 2=2-1", payload, flags=re.IGNORECASE)


def sqli_version_comment(payload: str, _rng: random.Random | None = None) -> str:
    """MySQL 조건부 실행 주석으로 키워드 감싸기: `UNION`→`/*!50000UNION*/`."""
    return re.sub(r"\b(UNION|SELECT)\b", r"/*!50000\1*/", payload, flags=re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 4.3 XSS 전용
# ─────────────────────────────────────────────────────────────────────────────

def xss_html_entity(payload: str, _rng: random.Random | None = None) -> str:
    """핵심 특수문자를 HTML 엔티티로: `<`→`&lt;`, `>`→`&gt;`(브라우저가 환원)."""
    return payload.replace("<", "&lt;").replace(">", "&gt;")


def xss_decimal_entity(payload: str, _rng: random.Random | None = None) -> str:
    """`<`,`>` 를 10진 수치 참조로: `<`→`&#60;`, `>`→`&#62;`."""
    return payload.replace("<", "&#60;").replace(">", "&#62;")


def xss_tag_case(payload: str, _rng: random.Random | None = None) -> str:
    """스크립트 태그의 대소문자 변형: `<script>`→`<ScRiPt>`(HTML 태그 대소문자 무관)."""
    def flip(m: re.Match) -> str:
        return "".join(c.upper() if i % 2 else c.lower()
                       for i, c in enumerate(m.group(0)))
    return re.sub(r"script", flip, payload, flags=re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 4.4 Command Injection 전용
# ─────────────────────────────────────────────────────────────────────────────

def cmdi_quote_insert(payload: str, _rng: random.Random | None = None) -> str:
    """셸 명령어 사이에 빈 인용 삽입: `cat`→`c""at`(셸이 빈 문자열 제거·동일 실행)."""
    for cmd in ("cat", "ls", "id", "whoami", "wget", "curl"):
        payload = re.sub(rf"\b{cmd}\b", cmd[0] + '""' + cmd[1:], payload)
    return payload


def cmdi_ifs_substitution(payload: str, _rng: random.Random | None = None) -> str:
    """명령어 뒤 공백을 `${IFS}` 로 치환(셸 내부 필드 구분자 — 공백과 동치)."""
    return re.sub(r"(\b(?:cat|ls|id|whoami|wget|curl)\b) ", r"\1${IFS}", payload)


def cmdi_separator_swap(payload: str, _rng: random.Random | None = None) -> str:
    """명령 구분자 치환: `;`→`|`(둘 다 명령 연쇄, 문맥에 따라 동치)."""
    return payload.replace(";", "|")


# ─────────────────────────────────────────────────────────────────────────────
# 등록제 — 이름→함수, 그리고 클래스별 적용 가능 변형
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY = {
    # 공통
    "url_encode": url_encode,
    "double_url_encode": double_url_encode,
    "random_case": random_case,
    "space_to_comment": space_to_comment,
    "space_to_tab": space_to_tab,
    # SQLi
    "sqli_inline_comment": sqli_inline_comment,
    "sqli_logical_equiv": sqli_logical_equiv,
    "sqli_version_comment": sqli_version_comment,
    # XSS
    "xss_html_entity": xss_html_entity,
    "xss_decimal_entity": xss_decimal_entity,
    "xss_tag_case": xss_tag_case,
    # CmdI
    "cmdi_quote_insert": cmdi_quote_insert,
    "cmdi_ifs_substitution": cmdi_ifs_substitution,
    "cmdi_separator_swap": cmdi_separator_swap,
}

COMMON = ["url_encode", "double_url_encode", "random_case", "space_to_comment", "space_to_tab"]

# 클래스명(전처리 라벨과 일치) → 그 클래스에 의미보존이 성립하는 변형 목록(공통 + 전용)
CLASS_MUTATIONS = {
    "SQLInjection": COMMON + ["sqli_inline_comment", "sqli_logical_equiv", "sqli_version_comment"],
    "XSS": COMMON + ["xss_html_entity", "xss_decimal_entity", "xss_tag_case"],
    "CommandInjection": COMMON + ["cmdi_quote_insert", "cmdi_ifs_substitution", "cmdi_separator_swap"],
}


def apply_mutation(name: str, payload: str, rng: random.Random | None = None) -> str:
    """이름으로 단일 변형을 적용한다."""
    if name not in REGISTRY:
        raise KeyError(f"알 수 없는 변형: {name} (가능: {sorted(REGISTRY)})")
    return REGISTRY[name](payload, rng)


def apply_chain(names: list[str], payload: str, rng: random.Random | None = None) -> str:
    """여러 변형을 순차 적용한다(조합/예산 공격용). 앞에서부터 차례로 덧씌운다."""
    for name in names:
        payload = apply_mutation(name, payload, rng)
    return payload
