"""
Phase 4 — 얕은 CNN (캐스케이드 1차 = RGB CNN, 단독으로는 지표 비교 기준) (페이로드 이미지 분류)

설계 문서 5장 1차 모델 구현:
    Conv(32) → Conv(64) → Conv(128), 각 블록에 BatchNorm + ReLU + MaxPool,
    마지막에 Global Average Pooling(GAP) → FC → Softmax(K-class).

설계 의도:
    - "얕은" 구조: 페이로드 이미지는 48×48 소형이고 상단 일부에 신호가 몰려 있어
      깊은 망은 과적합/과설계다. 3블록이면 충분하다는 판단(설계 4·5장).
    - GAP 사용: FC 파라미터를 줄여 과적합을 억제하고 입력 크기(32/48/64) 변화에도
      코드 수정 없이 대응(ablation 용이).
    - BatchNorm: 바이트값 분포가 클래스마다 달라 학습 안정화에 유리.

torch 는 이 파일에서만 import 한다(RGB CNN 전용 모듈).
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


# ---------------------------------------------------------------------------
# Phase 12 (M4) — 다단 조기종료 CNN
# ---------------------------------------------------------------------------
# 보조 헤드 손실 가중치 기본값. 너무 크면 보조 헤드가 본 헤드 학습을 방해하고(docs/11 §9 리스크),
# 너무 작으면 보조 헤드가 못 배워 조기종료가 일어나지 않는다. 0.3 은 그 사이의 관례값이다.
DEFAULT_AUX_WEIGHT = 0.3
# 조기종료 임계값 기본값. **학습이 아니라 추론 시점의 손잡이**라 체크포인트와 무관하다
# (캐스케이드의 τ 와 같은 성격 — val 에서 고르고 test 에 1회 적용한다).
DEFAULT_EXIT_THRESHOLD = 0.99
# 1 보다 큰 임계값 = "아무도 조기종료하지 않음" = 일반 3블록 CNN 과 정확히 같은 동작.
# 학습·체크포인트 선택은 이 값으로 한다(임계값이 정해지기 전이라 아키텍처의 상한을 봐야 한다).
NO_EARLY_EXIT = 1.01


class EarlyExitCNN(nn.Module):
    """블록마다 보조 분류기를 단 조기종료 CNN (캐스케이드 1차 비용 C_cnn 자체를 낮춘다).

    왜 필요한가 (docs/11 §5.1):
        캐스케이드 비용식 C(r) = C_cnn + r·C_charcnn 에서 지금까지 건드린 것은 r 뿐이었다.
        게이트를 완벽히 고쳐도(oracle r=1.87%) 7.29배가 천장인데, 그 원인은 C_cnn 이
        바닥이기 때문이다. 1차 내부에서 쉬운 트래픽을 첫 블록에서 확정하면 C_cnn 이 내려가고,
        천장 자체가 올라간다. 게이트 개선(F6)과 **직교**하며 이득이 곱해진다.

    백본은 PayloadCNN 과 **완전히 동일**하다(Conv 32→64→128 + GAP). 그래야 "조기종료를
    붙였을 때의 순수 효과"만 분리해 비교할 수 있다. 늘어나는 파라미터는 보조 헤드 두 개뿐이다.

    ⚠️ 학습 때는 조기종료를 하지 않는다:
        모든 헤드가 모든 샘플을 봐야 학습이 된다. 또한 BatchNorm 이 train 모드에서는 배치
        통계를 쓰므로, 샘플을 중간에 빼면 배치 구성에 따라 결과가 달라진다(비결정성).
        eval 모드에서는 running stats 를 쓰므로 부분집합만 진행해도 샘플별 결과가 같다 —
        조기종료를 **실제 연산 생략**으로 구현할 수 있는 근거가 이것이다.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3,
                 exit_threshold: float = DEFAULT_EXIT_THRESHOLD):
        super().__init__()

        def conv_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # PayloadCNN 의 features 와 같은 3블록. ModuleList 로 쪼갠 이유는 블록 사이에서
        # 빠져나갈 수 있어야 하기 때문이다(nn.Sequential 은 중간 개입이 안 된다).
        self.blocks = nn.ModuleList([
            conv_block(in_channels, 32),   # 48 → 24
            conv_block(32, 64),            # 24 → 12
            conv_block(64, 128),           # 12 → 6
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        # 헤드는 블록 출력 채널 수에 맞춘다. 마지막이 본 헤드(= PayloadCNN.classifier 와 동형).
        self.heads = nn.ModuleList([
            nn.Linear(32, num_classes),
            nn.Linear(64, num_classes),
            nn.Linear(128, num_classes),
        ])

        # ⚠️ 아래 둘은 **일반 속성**이다(buffer/parameter 로 등록하지 않는다).
        #    등록하면 state_dict 에 들어가, 임계값만 다른 실행이 체크포인트 호환성을 깨뜨린다.
        #    임계값은 추론 손잡이지 학습된 값이 아니다.
        self.exit_threshold = float(exit_threshold)
        self.last_exit_stage = None  # 직전 forward 에서 각 샘플이 빠져나간 블록 인덱스(비용 회계용)

    @property
    def n_stages(self) -> int:
        return len(self.blocks)

    def _head(self, feat: torch.Tensor, i: int) -> torch.Tensor:
        """블록 출력 → 해당 단계의 logits. 본 헤드에만 dropout 을 건다(PayloadCNN 과 동형)."""
        z = torch.flatten(self.gap(feat), 1)
        if i == self.n_stages - 1:
            z = self.dropout(z)
        return self.heads[i](z)

    def forward_all_heads(self, x: torch.Tensor) -> list[torch.Tensor]:
        """모든 헤드의 logits 를 전량 계산한다(학습용 — 조기종료 없음)."""
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            out.append(self._head(x, i))
        return out

    @torch.no_grad()
    def forward_with_exits(self, x: torch.Tensor,
                           threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """조기종료를 **실제 연산 생략**으로 적용한다. 반환: (샘플별 logits, 빠져나간 단계).

        핵심: 확정된 샘플은 인덱싱으로 배치에서 빼고 남은 것만 다음 블록에 넣는다.
        마스킹으로 구현하면 배치 전체가 최대 깊이를 따라가 **시간이 전혀 줄지 않는다** —
        그러면 M4 의 지연 주장이 성립하지 않으므로, 여기서 축소 배치로 진행하는 것이 핵심이다.
        (캐스케이드가 2차를 에스컬레이션분에만 돌리는 것과 같은 구조다.)

        threshold ≤ 0 이면 전원이 1블록에서 확정(가장 싸고 부정확), > 1 이면 아무도 조기종료
        하지 않아 일반 3블록 CNN 과 정확히 같아진다(양 끝점이 두 극단 모델).
        """
        thr = self.exit_threshold if threshold is None else float(threshold)
        n = x.shape[0]
        num_classes = self.heads[-1].out_features
        logits_out = torch.zeros(n, num_classes, device=x.device, dtype=torch.float32)
        stage_out = torch.full((n,), -1, device=x.device, dtype=torch.long)
        if n == 0:
            return logits_out, stage_out

        active = torch.arange(n, device=x.device)  # 아직 확정되지 않은 샘플의 원본 인덱스
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            logits = self._head(h, i)

            if i == self.n_stages - 1:
                # 마지막 블록에서는 남은 전원이 확정된다.
                logits_out[active] = logits.float()
                stage_out[active] = i
                break

            done = torch.softmax(logits, dim=1).max(dim=1).values >= thr
            if done.any():
                idx = active[done]
                logits_out[idx] = logits[done].float()
                stage_out[idx] = i
            keep = ~done
            if not keep.any():
                break  # 전원이 여기서 확정 → 남은 블록은 아예 실행하지 않는다
            h = h[keep]
            active = active[keep]

        return logits_out, stage_out

    def forward(self, x: torch.Tensor):
        """학습 모드면 모든 헤드의 logits 리스트, 평가 모드면 '확정된 헤드'의 logits.

        평가 모드에서 단일 텐서를 돌려주는 덕분에, 기존 코드(train.evaluate,
        cascade.predict_probs, run_evasion 의 proba)가 **수정 없이** 그대로 동작한다.
        어느 단계에서 나갔는지는 last_exit_stage 에 남겨 비용 회계에 쓴다.
        """
        if self.training:
            return self.forward_all_heads(x)
        logits, stage = self.forward_with_exits(x)
        self.last_exit_stage = stage
        return logits


def build_early_exit_model(num_classes: int, **kwargs) -> EarlyExitCNN:
    """train.py 가 이름('cnn_ee')으로 조기종료 모델을 만들 수 있게 하는 팩토리 함수."""
    return EarlyExitCNN(num_classes=num_classes, **kwargs)


def exit_distribution(stages, n_stages: int) -> list[float]:
    """단계별 종료 비율 [p_stage0, p_stage1, ...]. 비용 회계와 보고용."""
    import numpy as np

    s = np.asarray(stages)
    if s.size == 0:
        return [0.0] * n_stages
    return [float((s == i).mean()) for i in range(n_stages)]
