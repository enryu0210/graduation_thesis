"""
Phase 3-B — 전처리된 CSV → 이미지 데이터셋(.npz) 빌드 스크립트

목적:
    data/processed/{track}_{split}.csv 를 읽어 각 페이로드를 (side,side) 이미지로 변환하고,
    학습에 바로 쓸 수 있는 압축 배열(.npz)로 data/images/ 에 저장한다.
    또한 클래스별 샘플 이미지를 그림으로 저장해(논문 그림용) 변환이 올바른지 눈으로 확인한다.

왜 PNG 수십만 장이 아니라 .npz 인가:
    - 20만 장의 개별 PNG 는 파일 수가 너무 많아 I/O 가 느리고 관리가 어렵다.
    - (N, side, side) uint8 배열 하나로 두면 로딩이 빠르고, 0-패딩이 많아 압축률도 좋다.

.npz 구성:
    - images  : (N, side, side) uint8
    - labels  : (N,) int    (정수 인코딩된 클래스)
    - classes : (K,) str    (labels 정수 → 클래스명 매핑, 정렬된 순서)

사용법:
    python src/imaging/build_image_dataset.py                      # payload_4class 전체 split
    python src/imaging/build_image_dataset.py --track csic_binary
    python src/imaging/build_image_dataset.py --text decoded --side 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 같은 폴더의 변환 모듈을 import (스크립트 직접 실행 시 이 폴더가 sys.path 에 들어옴)
from payload_to_image import DEFAULT_SIDE, payload_to_image, payload_to_rgb_image
from channel_encoders import DEFAULT_RGB_ENCODERS, ENCODERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "imaging"

SPLITS = ("train", "val", "test")


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    """문자열 라벨을 정수로 인코딩한다.

    반환: (정수 배열, 클래스명 리스트[인덱스=정수코드])
    정렬된 순서로 매핑해 실행마다 코드가 동일하도록(재현성) 한다.
    """
    classes = sorted(labels.unique())
    mapping = {name: i for i, name in enumerate(classes)}
    codes = labels.map(mapping).to_numpy(dtype=np.int64)
    return codes, classes


# 채널 인코더 이름 → 파일명용 짧은 약어(ablation 조합을 파일명으로 구분하기 위함).
_ENCODER_ABBR = {
    "raw_byte": "rb",
    "char_class": "cc",
    "normalized_char_class": "nc",
    "local_entropy": "le",
    "bit_popcount": "bp",
    "structural_special": "ss",
    "byte_delta": "bd",
}


def dataset_suffix(channels: str, encoders: tuple[str, str, str]) -> str:
    """저장 .npz 파일명에 붙는 채널 접미사를 만든다(gray/rgb + ablation 조합 구분).

    - gray            : "" (기존 파일명 그대로 — 하위호환)
    - rgb 기본 조합    : "_rgb"
    - rgb 커스텀 조합  : "_rgb-rb-cc-le" 처럼 채널 약어를 붙여 조합별로 파일이 안 겹치게 한다.
    이 규칙은 data_image.npz_path(로드측)와 반드시 일치해야 한다.
    """
    if channels == "gray":
        return ""
    if tuple(encoders) == tuple(DEFAULT_RGB_ENCODERS):
        return "_rgb"
    return "_rgb-" + "-".join(_ENCODER_ABBR.get(name, name) for name in encoders)


def build_split(
    track: str,
    split: str,
    text_col: str,
    side: int,
    channels: str,
    encoders: tuple[str, str, str],
) -> tuple[Path, int]:
    """한 split CSV 를 이미지 배열(.npz)로 변환·저장한다. 반환: (저장경로, 샘플 수).

    channels="gray"  → (N, side, side) uint8 그레이스케일
    channels="rgb"   → (N, side, side, 3) uint8 RGB (encoders = R,G,B 채널 인코더 이름)
    """
    csv_path = PROCESSED_DIR / f"{track}_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"전처리 파일이 없습니다: {csv_path} (먼저 preprocess.py 실행)")

    df = pd.read_csv(csv_path, encoding="utf-8")
    texts = df[text_col].fillna("").astype(str)

    # 성능을 위해 결과 배열을 미리 할당하고 한 행씩 채운다.
    n = len(df)
    if channels == "gray":
        images = np.zeros((n, side, side), dtype=np.uint8)
        for i, text in enumerate(texts):
            images[i] = payload_to_image(text, side=side)
    else:
        images = np.zeros((n, side, side, 3), dtype=np.uint8)
        for i, text in enumerate(texts):
            images[i] = payload_to_rgb_image(text, side=side, encoders=encoders)

    codes, classes = encode_labels(df["label"])

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = dataset_suffix(channels, encoders)
    out_path = IMAGES_DIR / f"{track}_{split}_{text_col}_{side}{suffix}.npz"
    # 채널 인코더 조합도 메타데이터로 함께 저장(재현성·해석용).
    np.savez_compressed(
        out_path,
        images=images,
        labels=codes,
        classes=np.array(classes),
        channel_encoders=np.array(encoders if channels == "rgb" else ["raw_byte"]),
    )
    return out_path, n


def save_sample_figure(
    track: str,
    text_col: str,
    side: int,
    channels: str,
    encoders: tuple[str, str, str],
) -> Path | None:
    """train split 에서 클래스별로 1장씩 뽑아 그리드 그림으로 저장한다(변환 검증·논문 그림용).

    rgb 모드에서는 각 클래스에 대해 RGB 합성 이미지 + R/G/B 채널 분해를 함께 보여줘
    "어떤 채널이 어떤 신호를 담는지"를 눈으로 확인할 수 있게 한다(교수 요구 대응).
    """
    csv_path = PROCESSED_DIR / f"{track}_train.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    classes = sorted(df["label"].unique())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = dataset_suffix(channels, encoders)

    if channels == "gray":
        fig, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 3.2))
        if len(classes) == 1:  # subplots 는 클래스가 1개면 배열이 아니라 단일 객체를 반환
            axes = [axes]
        for ax, cls in zip(axes, classes):
            sample_text = df[df["label"] == cls][text_col].fillna("").astype(str).iloc[0]
            img = payload_to_image(sample_text, side=side)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{cls}\n({side}x{side})", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{track} sample payloads as {side}x{side} grayscale ({text_col})", fontsize=11)
    else:
        # 행: [RGB 합성, R, G, B],  열: 클래스
        row_labels = ["RGB", f"R={encoders[0]}", f"G={encoders[1]}", f"B={encoders[2]}"]
        fig, axes = plt.subplots(4, len(classes), figsize=(3 * len(classes), 11))
        axes = np.atleast_2d(axes)
        if len(classes) == 1:
            axes = axes.reshape(4, 1)
        for col, cls in enumerate(classes):
            sample_text = df[df["label"] == cls][text_col].fillna("").astype(str).iloc[0]
            rgb = payload_to_rgb_image(sample_text, side=side, encoders=encoders)
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f"{cls}", fontsize=9)
            for ch in range(3):
                axes[ch + 1, col].imshow(rgb[:, :, ch], cmap="gray", vmin=0, vmax=255)
            for row in range(4):
                axes[row, col].axis("off")
                if col == 0:  # 왼쪽 첫 열에 채널 이름을 라벨로
                    axes[row, col].text(-0.15, 0.5, row_labels[row], rotation=90,
                                        va="center", ha="center", fontsize=9,
                                        transform=axes[row, col].transAxes)
        fig.suptitle(f"{track} RGB payloads {side}x{side} ({text_col}) "
                     f"R/G/B={'/'.join(encoders)}", fontsize=11)

    fig.tight_layout()
    out_path = FIG_DIR / f"{track}_samples_{text_col}_{side}{suffix}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="전처리 CSV → 이미지(.npz) 빌드")
    parser.add_argument("--track", default="payload_4class",
                        choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"],
                        help="변환할 데이터 트랙")
    parser.add_argument("--text", default="raw", choices=["raw", "decoded"],
                        help="이미지화할 텍스트 컬럼 (raw=원본, decoded=디코딩본)")
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE,
                        help=f"정사각 이미지 한 변(픽셀), 기본 {DEFAULT_SIDE}")
    parser.add_argument("--channels", default="gray", choices=["gray", "rgb"],
                        help="gray=1채널(기존), rgb=3채널(교수 요구: 채널 강화)")
    parser.add_argument("--rgb-encoders", default=",".join(DEFAULT_RGB_ENCODERS),
                        help="rgb 모드의 R,G,B 채널 인코더 이름(콤마 구분). "
                             f"사용 가능: {', '.join(sorted(ENCODERS))}. "
                             f"기본: {','.join(DEFAULT_RGB_ENCODERS)}")
    args = parser.parse_args()

    text_col = "text_raw" if args.text == "raw" else "text_decoded"

    # rgb 채널 인코더 파싱·검증(잘못된 이름이면 여기서 명확히 실패).
    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    if args.channels == "rgb":
        if len(encoders) != 3:
            parser.error(f"--rgb-encoders 는 R,G,B 3개여야 합니다: {args.rgb_encoders}")
        for name in encoders:
            if name not in ENCODERS:
                parser.error(f"알 수 없는 채널 인코더 '{name}'. 사용 가능: {', '.join(sorted(ENCODERS))}")

    print(f"=== 이미지 빌드: track={args.track}, text={args.text}, side={args.side}, "
          f"channels={args.channels}" + (f", RGB={encoders}" if args.channels == "rgb" else "") + " ===")
    total = 0
    for split in SPLITS:
        try:
            out_path, n = build_split(args.track, split, text_col, args.side, args.channels, encoders)
        except FileNotFoundError as exc:
            print(f"  [건너뜀] {split}: {exc}")
            continue
        total += n
        print(f"  [OK] {split}: {n:,} 장 → {out_path.relative_to(PROJECT_ROOT)}")

    fig_path = save_sample_figure(args.track, text_col, args.side, args.channels, encoders)
    if fig_path is not None:
        print(f"  [그림] 클래스별 샘플 → {fig_path.relative_to(PROJECT_ROOT)}")

    print(f"[완료] 총 {total:,} 장 변환.")


if __name__ == "__main__":
    main()
