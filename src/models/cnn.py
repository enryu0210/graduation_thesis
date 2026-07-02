"""
Phase 4 — 제안 모델: 얕은 CNN (페이로드 이미지 분류)

설계 문서 5장 "제안 모델" 구현:
    Conv(32) → Conv(64) → Conv(128), 각 블록에 BatchNorm + ReLU + MaxPool,
    마지막에 Global Average Pooling(GAP) → FC → Softmax(K-class).

설계 의도:
    - "얕은" 구조: 페이로드 이미지는 48×48 소형이고 상단 일부에 신호가 몰려 있어
      깊은 망은 과적합/과설계다. 3블록이면 충분하다는 판단(설계 4·5장).
    - GAP 사용: FC 파라미터를 줄여 과적합을 억제하고 입력 크기(32/48/64) 변화에도
      코드 수정 없이 대응(ablation 용이).
    - BatchNorm: 바이트값 분포가 클래스마다 달라 학습 안정화에 유리.

torch 는 이 파일에서만 import 한다(제안 CNN 전용 모듈).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PayloadCNN(nn.Module):
    """페이로드 그레이스케일 이미지(1채널)를 K개 클래스로 분류하는 얕은 CNN."""

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()

        # 한 블록 = Conv(3x3, padding=1) → BN → ReLU → MaxPool(2). 공간 크기를 절반으로 줄인다.
        def conv_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 32),   # 48 → 24
            conv_block(32, 64),            # 24 → 12
            conv_block(64, 128),           # 12 → 6
        )

        # GAP: (N,128,H,W) → (N,128,1,1). 공간 크기와 무관하게 128차원 특징으로 요약.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)   # (N,128,1,1) → (N,128)
        x = self.dropout(x)
        return self.classifier(x)  # logits (Softmax 는 손실함수/평가에서 적용)


def build_model(num_classes: int, **kwargs) -> PayloadCNN:
    """train.py 가 이름으로 모델을 만들 수 있게 하는 팩토리 함수."""
    return PayloadCNN(num_classes=num_classes, **kwargs)
