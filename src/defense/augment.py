"""
Phase 11 (RQ3) — 적대적 증강 데이터셋 생성 (docs/10 §4, §7.2)

역할:
    RQ2 의 의미보존 변형(src/attacks/mutations.py)을 **학습 데이터에 섞어** 방어 모델을
    만든다. 이 모듈은 "문자열 → 문자열" 순수 변환만 담당하고, 이미지화/시퀀스화는
    호출부(train.py)가 기존 파이프라인으로 처리한다.
    → docs/05 §6 의 "같은 파이프라인, 입력만 바꾼다" 원칙을 방어 쪽에도 그대로 적용.

두 가지 핵심 설계 (둘 다 docs/10 에서 사전 확정):

1) **계열 분할(seen / held-out)** — §4
   학습에 쓴 변형으로 평가하면 "본 적 있는 공격을 막았다"는 자명한 결과가 된다.
   변형 14종을 4계열로 묶고, 시나리오 S-A 는 가장 강력한 **인코딩 계열(E)을 통째로
   학습에서 제외**한다. 평가는 E 를 포함한 전량으로 하므로 "미지의 변형에 일반화되는가"를
   정면으로 묻는다.

2) **치환(replace) 방식** — §7.2
   변형본을 '추가'하면 공격 클래스만 불어나 Normal 이 상대적 소수가 되고, FPR·
   benign-evasion 이 방어 효과가 아니라 **분포 변화** 때문에 움직인다(--balance 전제도 깨짐).
   그래서 원본 샘플을 변형본으로 **바꿔치기**한다 → train 크기·클래스 비율 불변.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "attacks"))

import mutations as MU  # noqa: E402

# ---------------------------------------------------------------------------
# 변형 계열 (docs/10 §4 표) — 실측 위력이 아니라 '변형이 무엇을 건드리는가'로 묶었다.
# ---------------------------------------------------------------------------
FAMILIES: dict[str, tuple[str, ...]] = {
    # E: 표면 바이트를 %XX/&#NN 로 치환 — docs/05 §12.2 실측에서 any-misclass 를 거의 홀로 설명
    "E": ("url_encode", "double_url_encode", "xss_html_entity", "xss_decimal_entity"),
    # C: 대소문자만 흔든다
    "C": ("random_case", "xss_tag_case"),
    # W: 공백·주석 등 토큰 경계를 흔든다
    "W": ("space_to_tab", "space_to_comment", "sqli_inline_comment",
          "sqli_version_comment", "cmdi_ifs_substitution"),
    # S: 구문적으로 동치인 다른 표현으로 바꾼다
    "S": ("sqli_logical_equiv", "cmdi_quote_insert", "cmdi_separator_swap"),
}

# 학습에 쓰는 계열 조합. 평가는 항상 전량(E 포함)으로 한다.
MUTATION_SPLITS: dict[str, tuple[str, ...]] = {
    "S0": ("E", "C", "W", "S"),   # seen — 낙관 상한
    "SA": ("C", "W", "S"),        # held-out(E 제외) — 진짜 질문
}


def _validate_families() -> None:
    """모든 변형이 정확히 한 계열에 속하는지 확인한다(import 시점 자체 점검).

    왜 여기서 죽이나: 새 변형을 mutations.py 에 추가하고 계열 분류를 잊으면, 그 변형은
    어느 split 에도 안 들어가 **조용히 학습에서 빠진다**. 조용한 실패보다 즉시 실패가 낫다.
    """
    listed = [name for names in FAMILIES.values() for name in names]
    if len(listed) != len(set(listed)):
        dupes = sorted({n for n in listed if listed.count(n) > 1})
        raise RuntimeError(f"변형이 두 계열에 중복 배치됐습니다: {dupes}")
    missing = sorted(set(MU.REGISTRY) - set(listed))
    unknown = sorted(set(listed) - set(MU.REGISTRY))
    if missing or unknown:
        raise RuntimeError(
            "FAMILIES 가 mutations.REGISTRY 와 어긋납니다 "
            f"(계열 미분류: {missing} / 존재하지 않는 변형: {unknown}). "
            "변형을 추가했다면 augment.FAMILIES 에도 분류를 넣으세요."
        )


_validate_families()


def split_techniques(mutation_split: str) -> list[str]:
    """분할 이름 → 학습에 허용된 변형 이름 목록."""
    if mutation_split not in MUTATION_SPLITS:
        raise ValueError(f"알 수 없는 변형 분할: {mutation_split} (가능: {sorted(MUTATION_SPLITS)})")
    return [name for fam in MUTATION_SPLITS[mutation_split] for name in FAMILIES[fam]]


def held_out_techniques(mutation_split: str) -> list[str]:
    """그 분할에서 **평가 전용으로 남겨둔** 변형 목록(누수 검사·리포트용)."""
    seen = set(split_techniques(mutation_split))
    return sorted(set(MU.REGISTRY) - seen)


def augment_texts(texts: list[str], labels: list[str], *, mutation_split: str = "S0",
                  ratio: float = 0.5, budget: int = 3, seed: int = 42):
    """공격 샘플의 `ratio` 비율을 변형본으로 **치환**한다.

    인자:
        texts   : 원본 페이로드 문자열 리스트
        labels  : 같은 길이의 **문자열 라벨**(어떤 변형이 의미보존인지는 클래스가 정한다)
        ratio   : 클래스별 치환 비율(0~1). 0 이면 아무것도 안 바꾼다
        budget  : 한 샘플에 겹쳐 적용할 변형 개수 상한(1..budget 중 무작위)
        seed    : 결정론 보장용. 인덱스 선택(numpy)과 변형 적용(random) 모두 이 시드에서 파생

    반환:
        (augmented_texts, info)
        info = {"n_replaced", "per_label", "techniques", "held_out"}

    설계 메모:
        - Normal 처럼 의미보존 변형이 정의되지 않은 클래스는 **손대지 않는다**
          (mutations.CLASS_MUTATIONS 에 없는 라벨). 그래서 클래스 비율이 보존된다.
        - 변형 예산은 학습 1~budget, 평가는 k=5 까지 간다. 학습보다 강한 공격으로 평가해야
          낙관 편향이 없다(docs/10 §7.2).
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio 는 0~1 이어야 합니다: {ratio}")
    if budget < 1:
        raise ValueError(f"budget 은 1 이상이어야 합니다: {budget}")
    if len(texts) != len(labels):
        raise ValueError(f"texts({len(texts)}) 와 labels({len(labels)}) 길이가 다릅니다")

    allowed_all = split_techniques(mutation_split)
    out = list(texts)
    info = {"n_replaced": 0, "per_label": {}, "techniques": allowed_all,
            "held_out": held_out_techniques(mutation_split)}
    if ratio == 0.0:
        return out, info

    rng_idx = np.random.default_rng(seed)      # 어떤 샘플을 바꿀지
    rng_mut = random.Random(seed)              # 어떤 변형을 어떻게 적용할지

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[str(lab)].append(i)

    # sorted: 라벨 순회 순서가 시드와 함께 결과를 고정하도록(재현성)
    for label in sorted(by_label):
        # 그 클래스에 의미보존이 성립하면서 이번 분할에 허용된 변형만 남긴다
        usable = [t for t in allowed_all if t in MU.CLASS_MUTATIONS.get(label, [])]
        if not usable:
            continue  # Normal 등 — 변형 대상 아님
        idxs = by_label[label]
        n_pick = int(round(len(idxs) * ratio))
        if n_pick <= 0:
            continue
        chosen = rng_idx.choice(np.asarray(idxs), size=n_pick, replace=False)
        for i in sorted(int(x) for x in chosen):
            k = rng_mut.randint(1, min(budget, len(usable)))
            names = rng_mut.sample(usable, k)
            out[i] = MU.apply_chain(names, texts[i], rng_mut)
        info["per_label"][label] = n_pick
        info["n_replaced"] += n_pick

    return out, info
