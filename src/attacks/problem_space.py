"""
Phase 5 (RQ2) — Problem-space 회피 오케스트레이션 (변형 적용, 모델 무관)

목적:
    mutations.py 의 의미보존 변형을 **실제 공격 페이로드 집합**에 적용해, 회피 실험용
    변형 페이로드를 만든다. 이 파일은 '변형 적용'만 담당한다(모델 로딩·ASR 계산은
    run_evasion.py). docs/05 §4.5 의 "① 단일 기법 → ② 조합(예산)" 전략을 구현한다.

핵심 규칙(의미보존):
    변형은 **그 샘플의 클래스에 유효할 때만** 적용한다. 예: SQLi 전용 주석 삽입을 XSS
    페이로드에 걸면 의미가 깨질 수 있으므로, CLASS_MUTATIONS 매핑으로 걸러낸다.
    Normal 샘플은 애초에 공격이 아니므로 변형 대상이 아니다(applicable=False).
"""

from __future__ import annotations

import random

import numpy as np

import mutations as MU

# 공격 클래스(전처리 라벨 문자열 기준). Normal 은 회피 대상이 아니다.
ATTACK_LABELS = ["SQLInjection", "XSS", "CommandInjection"]

# 표시·순회 순서 고정용 전체 기법 목록(공통 5 + 클래스 전용 9).
ALL_TECHNIQUES = MU.COMMON + [
    "sqli_inline_comment", "sqli_logical_equiv", "sqli_version_comment",
    "xss_html_entity", "xss_decimal_entity", "xss_tag_case",
    "cmdi_quote_insert", "cmdi_ifs_substitution", "cmdi_separator_swap",
]


def applicable(technique: str, label: str) -> bool:
    """technique 이 그 클래스(label)에 의미보존이 성립하는 변형인가."""
    return technique in MU.CLASS_MUTATIONS.get(label, [])


def mutate_single(texts, labels, technique: str, rng: random.Random | None = None):
    """각 샘플에 단일 technique 를 적용한다(그 샘플 클래스에 유효할 때만).

    반환:
        mutated : list[str]  — 변형된(또는 미적용 시 원본) 텍스트
        applied : np.ndarray(bool) — 실제로 변형이 적용된 위치(=그 기법의 대상 공격 샘플)
    """
    rng = rng or random.Random(0)
    mutated, applied = [], []
    for t, lab in zip(texts, labels):
        if applicable(technique, lab):
            mutated.append(MU.apply_mutation(technique, t, rng))
            applied.append(True)
        else:
            mutated.append(t)
            applied.append(False)
    return mutated, np.asarray(applied)


def mutate_stacked(texts, labels, k: int, rng: random.Random | None = None):
    """각 공격 샘플에 그 클래스에 유효한 기법 중 k개를 랜덤 선택해 순차 적용한다.

    예산 k 를 키우며 호출하면 '예산-ASR 곡선'을 얻는다(docs/05 §4.5 ②).
    반환: (mutated, applied) — applied 는 공격 샘플(유효 기법이 있는 클래스) 위치.
    """
    rng = rng or random.Random(0)
    mutated, applied = [], []
    for t, lab in zip(texts, labels):
        techs = MU.CLASS_MUTATIONS.get(lab, [])
        if techs:
            chosen = rng.sample(techs, min(k, len(techs)))
            mutated.append(MU.apply_chain(chosen, t, rng))
            applied.append(True)
        else:
            mutated.append(t)
            applied.append(False)
    return mutated, np.asarray(applied)
