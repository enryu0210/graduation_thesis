"""얼린 RGB CNN의 GAP 표현을 확률·행 정렬 검증과 함께 저장한다."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models import cascade


@torch.no_grad()
def extract_loader_embeddings(net, loader, device):
    """dropout 전 표현을 사용해 고정 split과 CV의 정의를 일치시킨다."""
    net.eval()
    embeddings, probabilities = [], []
    for batch in loader:
        x = batch[0].to(device)
        emb = torch.flatten(net.gap(net.features(x)), 1)
        if emb.ndim != 2 or emb.shape[1] != 128:
            raise ValueError("CNN GAP 표현은 128차원이어야 합니다.")
        embeddings.append(emb.cpu().numpy().astype(np.float32))
        probabilities.append(torch.softmax(net.classifier(emb), 1).cpu().numpy())
    if not embeddings:
        raise ValueError("빈 입력에서는 표현을 추출할 수 없습니다.")
    emb, probs = np.concatenate(embeddings), np.concatenate(probabilities)
    if not np.isfinite(emb).all() or not np.isfinite(probs).all():
        raise ValueError("표현 또는 확률에 비유한 값이 있습니다.")
    return emb, probs


def validate_saved_probs(saved, arrays, probs):
    """라벨과 확률을 함께 검사해 같은 라벨 안에서의 행 뒤바뀜도 감지한다."""
    for split in ("val", "test"):
        if not np.array_equal(saved[f"y_{split}"], arrays[f"y_{split}"]):
            raise ValueError(f"{split} check-probs 라벨 정렬 불일치")
        reference = saved[f"p1_{split}"]
        if reference.shape != probs[split].shape or not np.allclose(
                reference, probs[split], atol=1e-4, rtol=0):
            raise ValueError(f"{split} check-probs 확률 정렬 불일치")
    if "classes" in saved and not np.array_equal(saved["classes"], arrays["classes"]):
        raise ValueError("check-probs 클래스 순서 불일치")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="srbh_4class", choices=("srbh_4class",))
    parser.add_argument("--channels", default="rgb", choices=("rgb",))
    parser.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--max-len", type=int, default=2304)
    parser.add_argument("--check-probs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.max_len <= 0:
        parser.error("--max-len은 양수여야 합니다.")
    try:
        encoders = tuple(part.strip() for part in args.rgb_encoders.split(","))
        tag = cascade.checkpoint_tag(args.track, "cnn", "raw", args.balance, args.channels, encoders)
        output = args.out or PROJECT_ROOT / f"experiments/results/emb_{tag}.npz"
        if output.exists():
            raise FileExistsError(f"기존 표현 파일을 덮어쓰지 않습니다: {output.name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arrays, probs = {}, {}
        net, classes = None, None
        for split in ("val", "test"):
            x, _, y, split_classes = cascade.build_inputs(
                args.track, split, "raw", 48, args.channels, encoders, args.max_len, None)
            if net is None:
                classes = list(split_classes)
                net = cascade.load_net("cnn", len(classes), args.track, "raw", args.balance,
                                       device, args.channels, 3, encoders)
                arrays["classes"] = np.asarray(classes)
            elif classes != list(split_classes):
                raise ValueError("val/test 클래스 순서 불일치")
            loader = DataLoader(TensorDataset(x), batch_size=512, shuffle=False)
            emb, reconstructed = extract_loader_embeddings(net, loader, device)
            expected = cascade.predict_probs(net, x, device, 512)
            if expected.shape != reconstructed.shape or not np.allclose(
                    reconstructed, expected, atol=1e-5, rtol=0):
                raise ValueError(f"{split} softmax(classifier(emb)) 일치 검사 실패")
            arrays[split], arrays[f"y_{split}"] = emb, y
            probs[split] = expected
            print(f"{split}: {emb.shape}, float32, softmax 일치 검사 통과", flush=True)
        if args.check_probs:
            with np.load(args.check_probs, allow_pickle=False) as saved:
                validate_saved_probs(saved, arrays, probs)
            print("check-probs: val/test 라벨·확률 행 정렬 검사 통과", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as saved:
            np.savez_compressed(saved, **arrays)
        print(f"저장: {output.name}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"표현 추출 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
