"""
Phase 11 — 산출물 tag 규칙 (단일 진실 소스)

왜 별도 모듈인가:
    tag 는 "어떤 실험인지"를 파일명으로 구분하는 유일한 장치다. 규칙이 여러 파일에
    복사돼 있으면 한 곳만 갱신됐을 때 **서로 다른 실험이 같은 파일을 덮어쓴다.**
    이 프로젝트는 그 사고를 이미 두 번 겪었다:
      - 커밋 5eede2f: RGB 채널 조합이 전부 '_rgb' 로 저장돼 상호 덮어쓰기
      - docs/09 §9.7: run_evasion 의 tag 에 운영점 τ 축이 없어 덮어쓸 구조였음
    Phase 11 에서 방어 축(defense/분할/증강비율)이 추가되면 규칙 사본이 4곳
    (train / cross_validate / cascade / run_evasion)이 되므로, 규칙 자체를 여기로 모은다.

tag 형식:
    {track}_{model}_{text}{채널}{패치}{lr}{방어}{_bal}
    예) payload_4class_csicnorm_cnn_raw_rgb_def-advtrain_SA_r0.5_bal
        payload_4class_cnn_raw                      (기본값만 쓰면 접미사 없음)

"기본값이면 접미사를 생략한다"는 관습을 지킨다 — 그래야 Phase 4~10 의 기존 산출물
파일명과 100% 호환되고, 과거 결과를 재생성 없이 계속 쓸 수 있다.
"""

from __future__ import annotations

import data_image

# 기본 학습률. "기본값이면 _lr 접미사 생략" 판정의 기준값이며, CNN 기준으로 검증된 값이다
# (단일 ViT 는 이 값에서 발산한다 — docs/08 §9.1).
DEFAULT_LR = 1e-3

# 적대적 증강 기본값. tag 와 CLI 기본값이 어긋나면 파일명이 실험을 잘못 표현하므로
# train.py 와 여기서 같은 상수를 본다.
DEFAULT_AUG_RATIO = 0.5
DEFAULT_AUG_BUDGET = 3

DEFENSE_MODES = ("none", "advtrain", "norm")


def defense_suffix(defense: str = "none", mutation_split: str | None = None,
                   aug_ratio: float | None = None) -> str:
    """방어 축 접미사.

    - none     : ""                          (기존 산출물과 파일명 호환)
    - norm     : "_def-norm"                 (입력 정규화는 학습 데이터를 안 바꿈 → 축이 없음)
    - advtrain : "_def-advtrain_{분할}_r{비율}"  (분할·비율이 다르면 완전히 다른 실험)
    """
    if defense in (None, "none"):
        return ""
    if defense == "norm":
        return "_def-norm"
    if defense == "advtrain":
        if mutation_split is None or aug_ratio is None:
            raise ValueError("advtrain 은 mutation_split 과 aug_ratio 가 필요합니다(tag 를 가르는 축)")
        return f"_def-advtrain_{mutation_split}_r{aug_ratio:g}"
    raise ValueError(f"알 수 없는 방어 방식: {defense} (가능: {DEFENSE_MODES})")


def build_tag(track: str, model: str, text: str = "raw", *,
              channels: str = "gray", encoders: tuple[str, str, str] | None = None,
              patch: str | None = None, lr: float | None = None,
              balance: bool = False, defense: str = "none",
              mutation_split: str | None = None, aug_ratio: float | None = None) -> str:
    """산출물/체크포인트 공통 tag 를 만든다.

    인자는 "이미 결정된 값"만 받는다(모델이 이미지인지 등의 판정은 호출부 책임).
    - channels/encoders : data_image._channel_suffix 를 그대로 재사용(저장·로드 파일명 일치)
    - patch             : ViT 전용. None 이면 생략
    - lr                : DEFAULT_LR 이거나 None 이면 생략
    - defense 계열      : defense_suffix 참조
    """
    ch_tag = data_image._channel_suffix(channels, encoders)
    patch_tag = f"_p{patch}" if patch else ""
    lr_tag = "" if (lr is None or lr == DEFAULT_LR) else f"_lr{lr:g}"
    def_tag = defense_suffix(defense, mutation_split, aug_ratio)
    bal_tag = "_bal" if balance else ""
    return f"{track}_{model}_{text}{ch_tag}{patch_tag}{lr_tag}{def_tag}{bal_tag}"
