"""RQ2 의미보존 변형(mutations.py) 스모크 테스트.

검증 관점(docs/05 §6.2):
    변형이 (1) 표면형을 실제로 바꾸고, (2) '의미보존'이 성립하도록 되돌릴 수 있는지
    (가역 변형은 왕복 복원, 구조 변형은 구조 보존)를 스모크로 확인한다. 라이브 실행
    검증은 불가하므로 규칙 수준 보장을 테스트로 갈음한다(한계는 논문 명시).
"""

import random
from html import unescape
from urllib.parse import unquote

import mutations as MU

SQLI = "admin' OR 1=1 UNION SELECT user, pass FROM users -- "
XSS = "<script>alert(1)</script>"
CMDI = "test; cat /etc/passwd"


def test_registry_and_class_maps_consistent():
    """CLASS_MUTATIONS 의 모든 이름이 REGISTRY 에 실제로 존재해야 한다."""
    for cls, names in MU.CLASS_MUTATIONS.items():
        for n in names:
            assert n in MU.REGISTRY, f"{cls} 의 변형 {n} 이 REGISTRY 에 없음"


def test_url_encode_roundtrip():
    """URL 인코딩은 디코딩하면 원문으로 환원(의미보존의 근거)."""
    assert unquote(MU.url_encode(SQLI)) == SQLI


def test_double_url_encode_roundtrip():
    """이중 인코딩은 두 번 디코딩하면 원문으로 환원."""
    assert unquote(unquote(MU.double_url_encode(SQLI))) == SQLI


def test_url_encode_actually_changes():
    """특수문자가 있는 페이로드는 인코딩으로 표면형이 실제로 바뀌어야 한다."""
    assert MU.url_encode(SQLI) != SQLI


def test_random_case_is_deterministic_with_seed():
    """같은 seed 면 결과가 같아야 한다(재현성)."""
    a = MU.random_case(SQLI, random.Random(42))
    b = MU.random_case(SQLI, random.Random(42))
    assert a == b


def test_random_case_preserves_semantics_ignoring_case():
    """대소문자만 바뀌므로 소문자로 접으면 원문과 같아야 한다(키워드 의미보존)."""
    out = MU.random_case(SQLI, random.Random(1))
    assert out.lower() == SQLI.lower()


def test_random_case_keeps_string_literals():
    """균형 잡힌 문자열 리터럴('...') 안의 값은 대소문자를 건드리지 않는다.

    주의: 주입 페이로드처럼 따옴표가 불균형이면 리터럴 안/밖 판정이 모호해진다
    (random_case docstring 참조). 여기서는 균형 리터럴로 보호 동작만 검증한다.
    """
    payload = "SELECT col FROM t WHERE name='Admin'"
    out = MU.random_case(payload, random.Random(3))
    assert "'Admin'" in out  # 따옴표 안 'Admin' 은 원형 유지


def test_space_to_comment_reversible_structure():
    """공백→/**/ 는 주석을 공백으로 되돌리면 원문 구조로 복원."""
    out = MU.space_to_comment(SQLI)
    assert " " not in out and out.replace("/**/", " ") == SQLI


def test_sqli_inline_comment_keeps_keywords():
    """인라인 주석 삽입 후 /**/ 를 제거하면 원 키워드 구조가 남아야 한다."""
    out = MU.sqli_inline_comment("UNION SELECT x FROM y")
    assert "/**/" in out
    assert out.replace("/**/", " ").replace("  ", " ").strip() == "UNION SELECT x FROM y"


def test_xss_html_entity_reversible():
    """HTML 엔티티 인코딩은 unescape 로 원문 복원(브라우저 환원과 동치)."""
    assert unescape(MU.xss_html_entity(XSS)) == XSS
    assert unescape(MU.xss_decimal_entity(XSS)) == XSS


def test_xss_tag_case_changes_but_preserves_lower():
    """script 태그 대소문자 변형: 표면형은 바뀌되 소문자로 접으면 동일."""
    out = MU.xss_tag_case(XSS)
    assert out != XSS and out.lower() == XSS.lower()


def test_cmdi_quote_insert_reversible():
    """빈 인용 삽입은 제거하면 원문 복원(셸이 빈 문자열을 무시하는 것과 동치)."""
    out = MU.cmdi_quote_insert(CMDI)
    assert out != CMDI
    assert out.replace('""', "") == CMDI


def test_cmdi_ifs_substitution():
    """명령어 뒤 공백이 ${IFS} 로 치환돼야 한다."""
    out = MU.cmdi_ifs_substitution("cat /etc/passwd")
    assert "cat${IFS}/etc/passwd" == out


def test_apply_chain_stacks_in_order():
    """조합 적용은 순차로 덧씌워지고, 단일 적용들과 동등해야 한다."""
    chain = ["sqli_inline_comment", "random_case"]
    out = MU.apply_chain(chain, SQLI, random.Random(7))
    step1 = MU.apply_mutation("sqli_inline_comment", SQLI)
    step2 = MU.apply_mutation("random_case", step1, random.Random(7))
    assert out == step2


def test_unknown_mutation_raises():
    """등록되지 않은 변형 이름은 명확한 에러를 내야 한다."""
    try:
        MU.apply_mutation("nope", SQLI)
        assert False, "예외가 발생해야 함"
    except KeyError:
        pass
