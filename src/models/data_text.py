"""
Phase 4 — 텍스트 데이터셋 로딩 유틸 (베이스라인 입력)

목적:
    RQ1 비교 대상인 "텍스트 기반 모델"(TF-IDF+ML, char-CNN, BiLSTM)은 이미지가 아니라
    원본/디코딩 페이로드 문자열을 입력으로 쓴다. 이 문자열을 data/processed CSV 에서
    로드하고, 라벨을 정수로 인코딩한다.

핵심:
    - 라벨 정수 인코딩은 이미지 트랙(build_image_dataset.py)과 "동일 규칙(정렬 순서)"을
      써야 두 입력이 같은 정수 코드를 공유한다. → 여기서도 sorted(unique) 로 매핑한다.
    - torch 불필요(순수 pandas). char-CNN/BiLSTM 용 시퀀스 인코딩은 별도 함수로 제공하되
      torch 없이 numpy 로 반환한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def csv_path(track: str, split: str) -> Path:
    return PROCESSED_DIR / f"{track}_{split}.csv"


def load_text_split(track: str, split: str, text: str = "raw"):
    """한 split 의 (texts, labels_str) 를 로드한다.

    반환:
        texts      : list[str]  (text_raw 또는 text_decoded)
        labels_str : pd.Series[str]  (원본 문자열 라벨)
    """
    path = csv_path(track, split)
    if not path.exists():
        raise FileNotFoundError(f"전처리 CSV 가 없습니다: {path} (먼저 preprocess.py 실행)")

    df = pd.read_csv(path, encoding="utf-8")
    text_col = "text_raw" if text == "raw" else "text_decoded"
    texts = df[text_col].fillna("").astype(str).tolist()
    return texts, df["label"]


def build_label_encoding(labels_str: pd.Series) -> tuple[np.ndarray, list[str]]:
    """문자열 라벨 → 정수 코드. 이미지 트랙과 동일하게 sorted(unique) 순서로 매핑한다."""
    classes = sorted(labels_str.unique())
    mapping = {name: i for i, name in enumerate(classes)}
    codes = labels_str.map(mapping).to_numpy(dtype=np.int64)
    return codes, classes


def encode_labels_with(labels_str: pd.Series, classes: list[str]) -> np.ndarray:
    """이미 정한 클래스 순서(train 기준)로 val/test 라벨을 인코딩한다.

    val/test 를 train 과 "같은 매핑"으로 인코딩해야 라벨 코드가 어긋나지 않는다.
    """
    mapping = {name: i for i, name in enumerate(classes)}
    return labels_str.map(mapping).to_numpy(dtype=np.int64)


# ── char-level 시퀀스 인코딩 (char-CNN / BiLSTM 베이스라인용) ──────────────────
#
# 문자를 정수 인덱스로 바꾼 뒤 고정 길이로 자르거나 패딩한다(이미지 변환과 같은 철학).
# 바이트(0~255)를 그대로 어휘로 쓰면 어휘 크기가 256+1(pad)로 단순하고, 이미지 트랙과
# "같은 바이트 관점"을 공유해 비교가 공정하다. → UTF-8 바이트 단위 인코딩을 채택한다.

PAD_INDEX = 0          # 0은 패딩 전용
VOCAB_SIZE = 257       # 0(pad) + 바이트값 0~255 를 1~256 으로 시프트


def encode_byte_sequence(text: str, max_len: int) -> np.ndarray:
    """문자열을 UTF-8 바이트 인덱스 시퀀스(고정 길이 max_len)로 인코딩한다.

    - 바이트값 b(0~255) → 인덱스 b+1 (0은 패딩 예약)
    - max_len 보다 길면 앞에서부터 자르고(truncation), 짧으면 뒤를 0(pad)으로 채운다.
      (이미지 변환의 truncation/zero-padding 규칙과 일치시켜 조건을 맞춘다.)
    """
    raw = text.encode("utf-8", errors="replace")[:max_len]
    seq = np.zeros(max_len, dtype=np.int64)
    if raw:
        seq[: len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
    return seq


def encode_byte_matrix(texts: list[str], max_len: int) -> np.ndarray:
    """여러 문자열을 (N, max_len) 정수 시퀀스 행렬로 인코딩한다."""
    matrix = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        matrix[i] = encode_byte_sequence(text, max_len)
    return matrix
