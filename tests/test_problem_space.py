"""RQ2 problem-space 오케스트레이션(problem_space.py) 테스트.

검증: 변형이 '그 클래스에 유효할 때만' 적용되는가(의미보존 보장), Normal 은 제외되는가,
단일/조합 적용의 반환 형태가 올바른가.
"""

import random

import problem_space as PS

TEXTS = ["admin' OR 1=1--", "<script>alert(1)</script>", "; cat /etc/passwd", "hello world"]
LABELS = ["SQLInjection", "XSS", "CommandInjection", "Normal"]


def test_common_technique_applies_to_all_attacks_not_normal():
    """공통 기법(url_encode)은 공격 3종에 적용되고 Normal 에는 적용되지 않는다."""
    mutated, applied = PS.mutate_single(TEXTS, LABELS, "url_encode", random.Random(1))
    assert applied.tolist() == [True, True, True, False]
    assert mutated[3] == "hello world"  # Normal 은 원본 유지


def test_class_specific_technique_only_its_class():
    """SQLi 전용 기법은 SQLi 샘플에만 적용된다(다른 클래스는 미적용)."""
    _, applied = PS.mutate_single(TEXTS, LABELS, "sqli_inline_comment", random.Random(1))
    assert applied.tolist() == [True, False, False, False]


def test_single_changes_applied_samples():
    """적용된 샘플은 실제로 표면형이 바뀌어야 한다."""
    mutated, applied = PS.mutate_single(TEXTS, LABELS, "url_encode", random.Random(1))
    for i, was in enumerate(applied):
        if was:
            assert mutated[i] != TEXTS[i]


def test_stacked_applies_k_techniques_to_attacks():
    """조합 적용은 공격 샘플에 대해 원본을 바꾸고 Normal 은 남긴다."""
    mutated, applied = PS.mutate_stacked(TEXTS, LABELS, 3, random.Random(1))
    assert applied.tolist() == [True, True, True, False]
    assert mutated[0] != TEXTS[0] and mutated[3] == TEXTS[3]


def test_stacked_k_zero_is_noop_for_attacks():
    """k=0 이면 조합할 기법이 없어 원본 그대로여야 한다(곡선의 baseline)."""
    mutated, _ = PS.mutate_stacked(TEXTS, LABELS, 0, random.Random(1))
    assert mutated == TEXTS


def test_all_techniques_are_registered():
    """ALL_TECHNIQUES 의 모든 이름이 mutations.REGISTRY 에 존재한다."""
    import mutations as MU
    for t in PS.ALL_TECHNIQUES:
        assert t in MU.REGISTRY
