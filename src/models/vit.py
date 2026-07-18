"""
Phase 9 — 제안 모델 arm ②: Vision Transformer (페이로드 이미지 분류)

목적:
    제안 CNN(cnn.py)과 **완전히 같은 입력**(48×48 gray 1채널 / RGB 3채널 .npz)을 받아
    같은 학습 루프·같은 지표로 평가되는 Transformer 계열 모델을 제공한다.
    즉 "이미지화 파이프라인은 그대로 두고 분류기만 CNN ↔ ViT 로 교체"하는 비교를 가능하게 한다.

두 가지 변형(둘 다 입력은 동일, 다른 건 아키텍처뿐):
    1) "vit"    — 단일 ViT. CNN stem 없이 패치 임베딩 + Transformer 블록만으로 구성.
                  순수하게 "지역성 귀납편향 없이도 되는가"를 본다.
    2) "hybrid" — CNN stem(conv 2블록) + Transformer. 지역성 편향을 남긴 채 전역 attention 을 얹는다.
                  데이터 20만 장 규모에서는 이쪽이 더 현실적인 성능 후보다(docs/08 §6).

왜 채널이 그대로 붙는가:
    ViT 의 패치 임베딩은 내부적으로 Conv2d(in_channels, dim, kernel=patch, stride=patch) 다.
    즉 채널을 섞는 지점이 CNN 의 첫 conv 와 동일하므로, Phase 7 의 RGB 채널 인코더
    (raw_byte / char_class / byte_delta …)가 **코드 수정 없이** 그대로 적용된다.

모델 용량 메모(참고):
    제안 CNN 93,988 → hybrid 약 0.6M → vit 약 2.7M.
    ViT 계열이 파라미터가 훨씬 크므로, 성능이 같다면 CNN 이 낫다는 해석을 잊지 말 것.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 기본 하이퍼파라미터(작은 데이터셋 20만 장 기준으로 "작게" 잡았다).
# 크게 잡으면 과적합·처리량 손해가 커서 이 arm 의 의미가 사라진다(docs/08 §6).
VIT_EMBED_DIM, VIT_DEPTH, VIT_HEADS = 192, 6, 3
HYBRID_EMBED_DIM, HYBRID_DEPTH, HYBRID_HEADS = 128, 4, 4

DEFAULT_PATCH = (8, 8)


def parse_patch(spec: str | tuple[int, int]) -> tuple[int, int]:
    """'8x8' / '1x48' 같은 문자열을 (높이, 너비) 튜플로 바꾼다.

    행 패치(1x48)를 쓰면 토큰 1개가 '연속된 48바이트'가 된다(docs/08 §2).
    정사각 패치(8x8)는 세로로 48바이트씩 건너뛴 조각들을 한 토큰에 섞는다.
    """
    if isinstance(spec, tuple):
        return spec
    try:
        h, w = spec.lower().split("x")
        return int(h), int(w)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"패치 형식이 잘못됐습니다: {spec!r} (예: '8x8', '1x48')") from exc


class PayloadViT(nn.Module):
    """단일 ViT — CNN stem 없이 패치 임베딩 + Transformer 블록만 사용한다.

    구현은 timm 의 VisionTransformer 에 위임한다. ViT 는 초기화 기법(truncated normal,
    pos-embed 스케일 등)에 성능이 민감해 직접 구현하면 미묘한 버그로 학습이 안 되는 일이 잦다.
    표준 구현을 쓰는 편이 비교의 공정성에도 유리하다.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, img_size: int = 48,
                 patch_size: str | tuple[int, int] = DEFAULT_PATCH, dropout: float = 0.1):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # 의존성 누락을 조용히 넘기지 않는다.
            raise ImportError(
                "ViT 모델에는 timm 이 필요합니다. `pip install timm` 후 다시 실행하세요."
            ) from exc

        ph, pw = parse_patch(patch_size)
        # 패치가 이미지 크기를 나누어떨어지지 않으면 timm 이 조용히 잘라내 정보가 사라진다.
        if img_size % ph or img_size % pw:
            raise ValueError(
                f"패치 {ph}x{pw} 가 이미지 {img_size}x{img_size} 를 나누어떨어지지 않습니다."
            )

        self.patch_size = (ph, pw)
        self.backbone = timm.models.VisionTransformer(
            img_size=img_size,
            patch_size=(ph, pw),
            in_chans=in_channels,     # gray=1 / rgb=3 이 그대로 들어온다
            num_classes=num_classes,
            embed_dim=VIT_EMBED_DIM,
            depth=VIT_DEPTH,
            num_heads=VIT_HEADS,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)   # logits (Softmax 는 손실함수/평가에서 적용)


class PayloadHybridViT(nn.Module):
    """CNN stem + Transformer — 지역성 편향을 남긴 채 전역 attention 을 얹는다.

    구조:
        conv 2블록 (48→24→12, 채널 64)  ← cnn.py 와 같은 블록 설계를 재사용
        → 12×12=144 토큰(각 64차원) → Linear 로 embed_dim 투영
        → 학습형 위치 임베딩 + Transformer 인코더 4층
        → 토큰 평균(GAP) → FC

    왜 cls 토큰 대신 토큰 평균인가:
        제안 CNN 이 GAP 로 요약하는 것과 동일한 방식이라 비교가 깔끔하고,
        작은 데이터셋에서 cls 토큰보다 학습이 안정적인 편이다.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, img_size: int = 48,
                 dropout: float = 0.1):
        super().__init__()

        def conv_block(cin: int, cout: int) -> nn.Sequential:
            # cnn.py 의 conv_block 과 동일 구성(비교 공정성을 위해 의도적으로 일치시킴).
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.stem = nn.Sequential(
            conv_block(in_channels, 32),   # 48 → 24
            conv_block(32, 64),            # 24 → 12
        )
        stem_out_ch = 64
        grid = img_size // 4               # conv 블록 2개 = 1/4 축소
        n_tokens = grid * grid

        self.proj = nn.Linear(stem_out_ch, HYBRID_EMBED_DIM)
        # 위치 임베딩: 토큰이 페이로드의 어느 구간인지 알려준다(앞=경로/파라미터명, 뒤=값).
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, HYBRID_EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HYBRID_EMBED_DIM,
            nhead=HYBRID_HEADS,
            dim_feedforward=HYBRID_EMBED_DIM * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (N, L, D) 순서 — 우리 데이터 흐름과 일치
            norm_first=True,    # pre-norm: 작은 데이터에서 학습이 더 안정적
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=HYBRID_DEPTH)
        self.norm = nn.LayerNorm(HYBRID_EMBED_DIM)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(HYBRID_EMBED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                       # (N,64,12,12)
        x = x.flatten(2).transpose(1, 2)       # (N,144,64) — 공간을 토큰 축으로
        x = self.proj(x) + self.pos_embed      # (N,144,D)
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)           # 토큰 평균(GAP 대응)
        return self.classifier(self.dropout(x))


def build_model(num_classes: int, in_channels: int = 1, variant: str = "vit",
                patch_size: str | tuple[int, int] = DEFAULT_PATCH, **kwargs) -> nn.Module:
    """train.py 가 이름으로 모델을 만들 수 있게 하는 팩토리(cnn.build_model 과 동일 규약)."""
    if variant == "vit":
        return PayloadViT(num_classes=num_classes, in_channels=in_channels,
                          patch_size=patch_size, **kwargs)
    if variant == "hybrid":
        # hybrid 는 conv stem 이 공간을 줄이므로 patch_size 개념이 없다(받되 무시).
        return PayloadHybridViT(num_classes=num_classes, in_channels=in_channels, **kwargs)
    raise ValueError(f"알 수 없는 ViT 변형: {variant} (vit | hybrid)")
