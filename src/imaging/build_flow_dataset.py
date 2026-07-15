"""
Phase 8(RQ4b) — pcap 흐름 → 이미지 데이터셋(.npz) 빌드 (악성/정상 이진 흐름 탐지)

목적:
    data/raw/ustc_tfc2016/{Benign,Malware}/*.pcap 을 세션 단위로 이미지화하고,
    RQ1 과 동일한 .npz 규격/파일명으로 data/images/ 에 저장한다.
    → train.py 가 track=ustc_flow_binary 로 그대로 로드해 같은 CNN·지표로 학습·평가.

과제 정의(정직성, docs/06):
    payload 4클래스(SQLi/XSS/CmdI/Normal)와 "다른 데이터·다른 단위(흐름)"다.
    따라서 "SQLi vs XSS 구분"이 아니라 **"악성 흐름 vs 정상 흐름"의 이진 탐지**로 명확히 서술한다.
    클래스명은 benign/malware — metrics.py 가 'benign' 을 정상으로 인식해 benign-evasion(악성→정상 누출)을
    자동 계산한다.

분할:
    세션 단위 stratified 70/15/15 (seed=42 결정론).
    ⚠️ 한계(논문 명시): 같은 호스트의 여러 세션이 train/test 에 섞일 수 있어 낙관적 편향 가능
    → 향후 host 단위 분할로 보강 여지. 파일럿 단계에서는 세션 단위로 진행.

사용법:
    python src/imaging/build_flow_dataset.py                     # grayscale, pcap당 상한 5000 세션
    python src/imaging/build_flow_dataset.py --channels rgb
    python src/imaging/build_flow_dataset.py --max-per-pcap 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from payload_to_image import DEFAULT_SIDE
from channel_encoders import DEFAULT_RGB_ENCODERS, ENCODERS
from flow_to_image import iter_session_bytes, session_to_image, session_to_rgb_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ustc_tfc2016"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "imaging"

TRACK = "ustc_flow_binary"
# 폴더명 → 이진 클래스명. sorted 시 benign=0, malware=1 (metrics 가 benign 을 정상으로 인식).
CATEGORY_TO_CLASS = {"Benign": "benign", "Malware": "malware"}
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test


def collect_images(channels: str, side: int, encoders: tuple[str, str, str],
                   max_per_pcap: int | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """모든 pcap 을 세션 이미지로 변환해 (images, labels, classes) 로 모은다."""
    classes = sorted(set(CATEGORY_TO_CLASS.values()))  # ['benign','malware']
    class_to_code = {name: i for i, name in enumerate(classes)}

    images_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for category, class_name in CATEGORY_TO_CLASS.items():
        pcap_dir = RAW_DIR / category
        # rglob: 일부 7z(SMB, Weibo)는 하위 폴더로 풀려 여러 pcap 으로 쪼개져 있음 → 재귀 탐색.
        pcaps = sorted(pcap_dir.rglob("*.pcap")) if pcap_dir.exists() else []
        if not pcaps:
            print(f"  [경고] pcap 없음: {pcap_dir} (download_ustc.py 먼저 실행)")
            continue
        for pcap_path in pcaps:
            count = 0
            for session_bytes in iter_session_bytes(pcap_path, max_bytes=side * side,
                                                    max_sessions=max_per_pcap):
                if channels == "gray":
                    img = session_to_image(session_bytes, side=side)
                else:
                    img = session_to_rgb_image(session_bytes, side=side, encoders=encoders)
                images_list.append(img)
                labels_list.append(class_to_code[class_name])
                count += 1
            print(f"  [{category}] {pcap_path.name}: {count:,} 세션")

    if not images_list:
        raise RuntimeError("세션 이미지가 하나도 없습니다. pcap 다운로드/경로를 확인하세요.")

    images = np.stack(images_list).astype(np.uint8)
    labels = np.array(labels_list, dtype=np.int64)
    return images, labels, classes


def stratified_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """클래스별 비율을 유지하며 train/val/test 인덱스로 나눈다(seed 고정, 결정론)."""
    rng = np.random.default_rng(seed)
    idx_by_split = {"train": [], "val": [], "test": []}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        idx_by_split["train"].append(idx[:n_train])
        idx_by_split["val"].append(idx[n_train:n_train + n_val])
        idx_by_split["test"].append(idx[n_train + n_val:])
    return {k: np.concatenate(v) for k, v in idx_by_split.items()}


def dataset_suffix(channels: str, encoders: tuple[str, str, str]) -> str:
    """build_image_dataset.dataset_suffix 와 동일 규칙(로더 파일명 일치)."""
    if channels == "gray":
        return ""
    if tuple(encoders) == tuple(DEFAULT_RGB_ENCODERS):
        return "_rgb"
    abbr = {"raw_byte": "rb", "char_class": "cc", "local_entropy": "le",
            "bit_popcount": "bp", "structural_special": "ss", "byte_delta": "bd"}
    return "_rgb-" + "-".join(abbr.get(n, n) for n in encoders)


def save_splits(images, labels, classes, splits, channels, side, encoders) -> None:
    """분할별로 .npz 를 data_image 로더가 읽는 규칙에 맞춰 저장한다."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = dataset_suffix(channels, encoders)
    for split, idx in splits.items():
        out = IMAGES_DIR / f"{TRACK}_{split}_text_raw_{side}{suffix}.npz"
        np.savez_compressed(
            out, images=images[idx], labels=labels[idx], classes=np.array(classes),
            channel_encoders=np.array(encoders if channels == "rgb" else ["raw_byte"]),
        )
        counts = np.bincount(labels[idx], minlength=len(classes))
        print(f"  [저장] {split}: {len(idx):,} 장 {dict(zip(classes, counts.tolist()))} → {out.name}")


def save_sample_figure(images, labels, classes, channels, side, encoders) -> Path:
    """클래스별 샘플 이미지를 그림으로 저장(변환 검증용)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 3.2))
    if len(classes) == 1:
        axes = [axes]
    for ax, (code, name) in zip(axes, enumerate(classes)):
        i = int(np.where(labels == code)[0][0])
        img = images[i]
        if channels == "gray":
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(f"{name}\n({side}x{side})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"USTC-TFC2016 flow sessions as {side}x{side} images", fontsize=11)
    fig.tight_layout()
    suffix = dataset_suffix(channels, encoders)
    out = FIG_DIR / f"{TRACK}_samples_text_raw_{side}{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="pcap 흐름 → 이미지(.npz) 빌드 (RQ4b)")
    parser.add_argument("--channels", default="gray", choices=["gray", "rgb"])
    parser.add_argument("--rgb-encoders", default=",".join(DEFAULT_RGB_ENCODERS),
                        help=f"rgb R,G,B 인코더. 사용 가능: {', '.join(sorted(ENCODERS))}")
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("--max-per-pcap", type=int, default=5000,
                        help="pcap(클래스 종류)당 세션 상한. 0/음수면 무제한. 기본 5000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    if args.channels == "rgb":
        if len(encoders) != 3 or any(n not in ENCODERS for n in encoders):
            parser.error(f"--rgb-encoders 는 유효한 R,G,B 3개여야 합니다: {args.rgb_encoders}")
    max_per_pcap = args.max_per_pcap if args.max_per_pcap and args.max_per_pcap > 0 else None

    print(f"=== 흐름 이미지 빌드: track={TRACK}, channels={args.channels}, "
          f"side={args.side}, max_per_pcap={max_per_pcap} ===")
    images, labels, classes = collect_images(args.channels, args.side, encoders, max_per_pcap)
    print(f"  총 세션 이미지: {len(images):,}  분포={dict(zip(classes, np.bincount(labels).tolist()))}")

    splits = stratified_split(labels, args.seed)
    save_splits(images, labels, classes, splits, args.channels, args.side, encoders)
    fig = save_sample_figure(images, labels, classes, args.channels, args.side, encoders)
    print(f"  [그림] 클래스별 샘플 → {fig.relative_to(PROJECT_ROOT)}")
    print(f"[완료] 학습: python src/models/train.py --model cnn --track {TRACK}"
          + (" --channels rgb" if args.channels == "rgb" else ""))


if __name__ == "__main__":
    main()
