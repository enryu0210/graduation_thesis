"""
Phase 4 — 이미지 데이터셋(.npz) 로딩 유틸 (제안 CNN 입력)

목적:
    Phase 3 이 만든 data/images/{track}_{split}_{text}_{side}.npz 를 읽어
    (1) numpy 로 바로 쓰거나, (2) PyTorch DataLoader 로 감싸 학습에 쓴다.

설계 원칙:
    - torch 는 "무거운 선택 의존성"이므로 파일 최상단에서 import 하지 않는다.
      numpy 로딩·클래스 가중치 계산 등 torch 없이 되는 기능은 torch 미설치 환경
      (예: 이 노트북)에서도 그대로 동작해야 한다. torch 관련 함수 안에서만 지연 import 한다.
    - 정규화(0~1)는 "저장은 uint8, 학습 직전 float"라는 Phase 3 결정을 따른다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "images"


def npz_path(track: str, split: str, text: str = "raw", side: int = 48) -> Path:
    """Phase 3 저장 규칙에 맞는 .npz 경로를 만든다."""
    return IMAGES_DIR / f"{track}_{split}_text_{text}_{side}.npz"


def load_split(track: str, split: str, text: str = "raw", side: int = 48):
    """한 split 의 (images, labels, classes) 를 로드한다.

    반환:
        images  : (N, side, side) uint8
        labels  : (N,) int64
        classes : list[str]  (정수 코드 → 클래스명)
    """
    path = npz_path(track, split, text, side)
    if not path.exists():
        raise FileNotFoundError(
            f"이미지셋이 없습니다: {path}\n"
            f"먼저 build_image_dataset.py 로 --track {track} --text {text} --side {side} 를 빌드하세요."
        )
    data = np.load(path, allow_pickle=False)
    images = data["images"]
    labels = data["labels"].astype(np.int64)
    classes = [str(c) for c in data["classes"]]
    return images, labels, classes


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """클래스 불균형 보정용 가중치(inverse frequency, 평균 1로 정규화)를 계산한다.

    공식: w_c = N / (K * count_c)  — 드문 클래스일수록 큰 가중치.
    설계 3장 "클래스 불균형 시 class weight" 요구사항 구현. CSIC(2-class)처럼
    한쪽으로 쏠린 트랙에서 특히 중요하다.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    # 빈 클래스가 있어도 0으로 나누지 않도록 최소 1로 클리핑.
    counts = np.clip(counts, 1.0, None)
    weights = labels.size / (n_classes * counts)
    return weights.astype(np.float32)


def make_torch_dataset(images: np.ndarray, labels: np.ndarray):
    """numpy (images, labels) 를 PyTorch TensorDataset 으로 감싼다.

    - 이미지: uint8(0~255) → float32(0~1)로 정규화하고 채널 축을 추가해 (N,1,H,W) 로 만든다.
      (Conv2d 는 (배치, 채널, 높이, 너비) 입력을 기대한다.)
    - torch 는 이 함수 안에서만 import 한다(지연 import).
    """
    import torch
    from torch.utils.data import TensorDataset

    # (N,H,W) uint8 → (N,1,H,W) float32(0~1)
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    y = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(x, y)


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int = 128, shuffle: bool = False):
    """이미지/라벨로 DataLoader 를 만든다(학습·평가 공용)."""
    from torch.utils.data import DataLoader

    dataset = make_torch_dataset(images, labels)
    # num_workers=0: Windows/Colab 어디서든 안전하게 동작(멀티프로세싱 이슈 회피).
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
