"""
Phase 4 — 베이스라인 ②③: char-level CNN, BiLSTM (RQ1 텍스트 딥러닝 비교군)

설계 문서 5장 "텍스트 딥러닝 베이스라인" 구현:
    페이로드를 이미지가 아니라 "바이트 시퀀스"로 처리하는 두 대표 계열.
    - CharCNN : 1D 컨볼루션으로 지역 n-gram 패턴을 잡는다.
    - BiLSTM  : 양방향 순환망으로 순차 의존성을 잡는다.

비교의 공정성:
    입력은 data_text.encode_byte_matrix 가 만든 (N, max_len) 바이트 인덱스 시퀀스로,
    제안 CNN 의 이미지 트랙과 "같은 바이트/같은 truncation·padding 규칙"을 공유한다.
    즉 "같은 원재료(바이트)를 2D 이미지로 보느냐 1D 시퀀스로 보느냐"만 다르게 해
    이미지화 자체의 효과를 분리해서 본다.

임베딩:
    바이트 인덱스(0=pad, 1~256=바이트값)를 학습형 임베딩으로 매핑한다.
    padding_idx=0 으로 두어 패딩 토큰이 학습에 영향을 주지 않게 한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data_text import PAD_INDEX, VOCAB_SIZE


class CharCNN(nn.Module):
    """바이트 임베딩 + 1D 다중 커널 CNN(문자 n-gram 탐지) 분류기."""

    def __init__(self, num_classes: int, embed_dim: int = 64,
                 num_filters: int = 128, kernel_sizes=(3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_INDEX)

        # 서로 다른 커널 크기 = 서로 다른 n-gram 폭. 결과를 이어 붙여 다양한 폭을 동시에 본다.
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L) 정수 → (N, L, E) → (N, E, L)  (Conv1d 는 채널 축이 가운데)
        emb = self.embedding(x).transpose(1, 2)
        # 각 커널로 컨볼루션 후 시간축 max-pooling → (N, num_filters)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        feat = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(feat))


class BiLSTM(nn.Module):
    """바이트 임베딩 + 양방향 LSTM 분류기."""

    def __init__(self, num_classes: int, embed_dim: int = 64,
                 hidden: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_INDEX)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # 양방향이라 hidden*2
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)              # (N, L, E)
        out, _ = self.lstm(emb)              # (N, L, 2*hidden)
        # 시간축 평균 풀링으로 시퀀스를 한 벡터로 요약(패딩 영향은 작게 유지).
        feat = out.mean(dim=1)
        return self.classifier(self.dropout(feat))


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """이름으로 텍스트 베이스라인 모델을 생성한다."""
    if name == "charcnn":
        return CharCNN(num_classes=num_classes, **kwargs)
    if name == "bilstm":
        return BiLSTM(num_classes=num_classes, **kwargs)
    raise ValueError(f"알 수 없는 텍스트 모델: {name} (charcnn | bilstm)")
