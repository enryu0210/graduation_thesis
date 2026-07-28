"""
Phase 4 — 공용 학습·평가 스크립트 (제안 CNN + 텍스트 베이스라인)

목적:
    제안 CNN(이미지)과 char-CNN/BiLSTM(바이트 시퀀스)을 "같은 학습 루프 / 같은 지표"로
    학습·평가한다. 모델·입력 형태만 다를 뿐 옵티마이저·조기종료·클래스 가중치·평가 방식은
    공유해야 RQ1 비교가 공정하다.

핵심 설계:
    - device 자동 감지(cuda 우선) → 이 노트북(CPU)에서도, Colab/4080(GPU)에서도 그대로 실행.
    - 손실: CrossEntropy(weight=class_weights) — 클래스 불균형 보정(설계 3장).
    - 조기 종료(early stopping): val Macro-F1 이 patience 에폭 동안 개선 없으면 중단하고
      best 가중치를 복원 → 과적합 방지 + 학습 시간 절약.
    - 재현성: 시드 고정. (완전 결정론까지는 강제하지 않되 실행 간 편차를 줄인다.)
    - 평가: 최종 test 셋 지표를 metrics.py 로 계산·저장(다른 베이스라인과 동일 포맷).

사용법 (Colab/GPU 권장):
    python src/models/train.py --model cnn                       # 제안 CNN
    python src/models/train.py --model charcnn                   # char-CNN 베이스라인
    python src/models/train.py --model bilstm                    # BiLSTM 베이스라인
    python src/models/train.py --model cnn --epochs 1 --limit 500 --smoke   # 스모크 테스트
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

# 이웃 모듈 import 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
# 적대적 증강(Phase 11)은 학습 시점에 '텍스트 → 이미지' 변환을 직접 호출하므로
# 이미지화 모듈과 방어 모듈 경로도 열어둔다(npz 에는 원문이 없다 — docs/10 §7.1).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "imaging"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "defense"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import metrics as M  # noqa: E402
import data_image  # noqa: E402
import data_text  # noqa: E402
import augment as AUG  # noqa: E402
from payload_to_image import payload_to_image, payload_to_rgb_image  # noqa: E402
from tagging import (  # noqa: E402
    DEFAULT_AUG_BUDGET, DEFAULT_AUG_RATIO, DEFAULT_LR, DEFENSE_MODES, build_tag,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "models"
CKPT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"

# 이미지를 쓰는 모델과 바이트 시퀀스를 쓰는 모델을 구분한다.
# vit/hybrid 는 제안 CNN 과 **완전히 같은 이미지 입력**을 받는 비교 arm 이다(vit.py 참조).
IMAGE_MODELS = {"cnn", "vit", "hybrid"}
TEXT_MODELS = {"charcnn", "bilstm"}

# DEFAULT_LR 등 tag 관련 상수는 tagging.py 가 단일 진실 소스다(위 import).
# 여기서 재노출하는 이유: cascade.py / cross_validate.py 가 `train.DEFAULT_LR` 로 참조 중이라
# 기존 import 경로를 깨지 않기 위함.


def set_seed(seed: int = 42) -> None:
    """실행 간 편차를 줄이기 위해 시드를 고정한다."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """가능하면 GPU(cuda)를, 없으면 CPU 를 쓴다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _limit(arr: np.ndarray, limit: int | None) -> np.ndarray:
    """스모크 테스트용: limit 이 주어지면 앞에서 그만큼만 사용한다."""
    return arr[:limit] if limit else arr


def balance_train_indices(y_train: np.ndarray, seed: int) -> np.ndarray:
    """훈련셋을 클래스 균형으로 언더샘플링할 인덱스를 만든다(train 전용).

    왜 필요한가 (RQ2 대비):
        payload_4class_csicnorm 은 Normal(≈3.4k) 이 공격(각 28~40k)보다 10배 이상 적다.
        이대로 학습하면 모델이 Normal 을 거의 예측하지 않게 되어, RQ2 의 핵심 지표인
        'benign-evasion(공격→Normal)' 이 인위적으로 어려워진다(=가짜 강건성). 이를 막기 위해
        각 클래스를 '가장 작은 클래스 수'에 맞춰 무작위 언더샘플링한다.

    왜 언더샘플링(오버샘플링 아님)인가:
        공격 클래스는 표본이 충분하므로, 소수 클래스에 맞춰 줄이면 데이터 중복 없이
        균형이 맞고 학습도 빨라진다. val/test 는 건드리지 않는다(실제 분포에서 평가해야 정직).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    per_class = min(int((y_train == c).sum()) for c in classes)
    keep = [rng.choice(np.where(y_train == c)[0], size=per_class, replace=False) for c in classes]
    keep = np.concatenate(keep)
    rng.shuffle(keep)  # 클래스별로 뭉치지 않도록 섞는다
    return keep


def texts_to_images(texts: list[str], side: int, channels: str,
                    encoders: tuple[str, str, str] | None) -> np.ndarray:
    """문자열 리스트 → (N,side,side[,3]) uint8 이미지 배열.

    build_image_dataset.py 가 .npz 를 만들 때 쓰는 것과 **완전히 같은 변환 함수**를 호출한다.
    ⚠️ 이 동일성이 깨지면 Phase 11 의 모든 수치가 무효다(원본은 npz, 증강본만 이 경로로
    만들면 두 경로의 미세한 차이가 '증강본 여부'와 상관돼 새 shortcut 이 된다 — docs/10 §7.1).
    그래서 증강 학습에서는 train split **전체**를 이 경로로 통일하고,
    tests/test_defense.py 가 npz 와 바이트 단위 일치를 검증한다.
    """
    if channels == "gray":
        return np.stack([payload_to_image(t, side=side) for t in texts])
    enc = tuple(encoders) if encoders else None
    return np.stack([payload_to_rgb_image(t, side=side, encoders=enc) for t in texts])


def _augment_split(texts: list[str], y: np.ndarray, classes: list[str],
                   mutation_split: str, ratio: float, budget: int, seed: int):
    """정수 라벨을 클래스명으로 되돌려 augment.augment_texts 를 호출하는 얇은 어댑터."""
    labels_str = [classes[int(c)] for c in y]
    return AUG.augment_texts(texts, labels_str, mutation_split=mutation_split,
                             ratio=ratio, budget=budget, seed=seed)


def build_datasets(model: str, track: str, text: str, side: int, max_len: int,
                   limit: int | None, balance: bool = False, seed: int = 42,
                   channels: str = "gray", encoders: tuple[str, str, str] | None = None,
                   defense: str = "none", mutation_split: str = "S0",
                   aug_ratio: float = DEFAULT_AUG_RATIO, aug_budget: int = DEFAULT_AUG_BUDGET):
    """모델 종류에 맞춰 (train/val/test TensorDataset, classes, class_weights, in_channels, aug_info) 를 만든다.

    - 이미지 모델: data/images 의 .npz → gray (N,1,H,W) / rgb (N,3,H,W) float
    - 텍스트 모델: data/processed 의 CSV → (N,L) 바이트 인덱스 시퀀스
    두 경로 모두 train 라벨로 클래스 수/가중치를 정한다.

    defense="advtrain" 이면 train/val 을 **CSV 원문에서** 만들고 의미보존 변형을 치환 주입한다:
      - train: 방어 학습 데이터
      - val  : 조기 종료 기준을 arm 의 목적함수에 맞추기 위해 같은 규칙으로 섞는다(docs/10 §7.3).
               ⚠️ held-out 계열은 train 에도 val 에도 들어가지 않는다(조기 종료를 통한 간접 누수 차단).
      - test : **절대 건드리지 않는다**(clean 평가). 회피 평가는 run_evasion.py 담당.
    defense="norm" 은 학습 데이터를 바꾸지 않는다 — `--text decoded` 로 입력 표현만 정규화한다.
    """
    advtrain = defense == "advtrain"
    aug_info: dict = {"train": None, "val": None}
    # 증강 순서: limit → balance → 치환. 균형화 이후에 바꿔야 클래스 비율이 보존된다(docs/10 §7.2).

    if model in IMAGE_MODELS:
        va_x, va_y, classes = data_image.load_split(track, "val", text, side, channels, encoders)
        te_x, te_y, _ = data_image.load_split(track, "test", text, side, channels, encoders)

        if advtrain:
            tr_txt, tr_lab = data_text.load_text_split(track, "train", text)
            # npz 의 클래스 순서로 인코딩한다(양쪽 다 sorted(unique) 규칙이라 동일해야 함).
            tr_y = data_text.encode_labels_with(tr_lab, classes)
            if limit:
                tr_txt, tr_y = tr_txt[:limit], tr_y[:limit]
            if balance:
                keep = balance_train_indices(tr_y, seed)
                tr_txt = [tr_txt[i] for i in keep]
                tr_y = tr_y[keep]
            tr_txt, aug_info["train"] = _augment_split(
                tr_txt, tr_y, classes, mutation_split, aug_ratio, aug_budget, seed)
            tr_x = texts_to_images(tr_txt, side, channels, encoders)

            # val 도 같은 규칙으로 섞는다. 시드를 다르게 줘 train 과 같은 변형 조합이
            # 그대로 재현되지 않게 한다(조기 종료가 train 변형에 과적합되는 것을 막음).
            va_txt, va_lab = data_text.load_text_split(track, "val", text)
            va_y_csv = data_text.encode_labels_with(va_lab, classes)
            if not np.array_equal(va_y_csv, va_y):
                # CSV 행 순서와 npz 행 순서가 어긋나면 이미지-라벨이 밀린다. 조용히 틀리느니 죽는다.
                raise RuntimeError(
                    "val 라벨이 CSV 와 npz 에서 불일치합니다 — 이미지 데이터셋을 다시 빌드하세요"
                    f"(build_image_dataset.py --track {track}).")
            va_txt, aug_info["val"] = _augment_split(
                va_txt, va_y, classes, mutation_split, aug_ratio, aug_budget, seed + 1)
            va_x = texts_to_images(va_txt, side, channels, encoders)
        else:
            tr_x, tr_y, _ = data_image.load_split(track, "train", text, side, channels, encoders)
            tr_x, tr_y = _limit(tr_x, limit), _limit(tr_y, limit)
            if balance:
                keep = balance_train_indices(tr_y, seed)
                tr_x, tr_y = tr_x[keep], tr_y[keep]

        train_ds = data_image.make_torch_dataset(tr_x, tr_y)
        val_ds = data_image.make_torch_dataset(va_x, va_y)
        test_ds = data_image.make_torch_dataset(te_x, te_y)
        y_train = tr_y
        # 이미지 채널 수(gray=1, rgb=3)를 배열 형태에서 직접 추론해 모델 in_channels 로 넘긴다.
        in_channels = 3 if (tr_x.ndim == 4 and tr_x.shape[-1] == 3) else 1

    elif model in TEXT_MODELS:
        tr_txt, tr_lab = data_text.load_text_split(track, "train", text)
        va_txt, va_lab = data_text.load_text_split(track, "val", text)
        te_txt, te_lab = data_text.load_text_split(track, "test", text)
        y_train, classes = data_text.build_label_encoding(tr_lab)
        y_val = data_text.encode_labels_with(va_lab, classes)
        y_test = data_text.encode_labels_with(te_lab, classes)

        if limit:
            tr_txt, y_train = tr_txt[:limit], y_train[:limit]
        if balance:
            keep = balance_train_indices(y_train, seed)
            tr_txt = [tr_txt[i] for i in keep]  # tr_txt 는 리스트라 컴프리헨션으로 인덱싱
            y_train = y_train[keep]

        if advtrain:
            tr_txt, aug_info["train"] = _augment_split(
                tr_txt, y_train, classes, mutation_split, aug_ratio, aug_budget, seed)
            va_txt, aug_info["val"] = _augment_split(
                va_txt, y_val, classes, mutation_split, aug_ratio, aug_budget, seed + 1)

        # 문자열 → (N,L) 바이트 시퀀스
        Xtr = data_text.encode_byte_matrix(tr_txt, max_len)
        Xva = data_text.encode_byte_matrix(va_txt, max_len)
        Xte = data_text.encode_byte_matrix(te_txt, max_len)
        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(y_val))
        test_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(y_test))
        in_channels = 1  # 텍스트 모델은 이미지 채널 개념이 없음(자리표시자)
    else:
        raise ValueError(f"알 수 없는 모델: {model}")

    class_weights = data_image.compute_class_weights(y_train, len(classes))
    return train_ds, val_ds, test_ds, classes, class_weights, in_channels, aug_info


def build_model(model: str, num_classes: int, in_channels: int = 1,
                patch: str = "8x8") -> nn.Module:
    """모델 이름 → nn.Module. 이미지 모델은 cnn.py/vit.py, 텍스트는 text_models.py 에서 가져온다."""
    if model == "cnn":
        import cnn
        return cnn.build_model(num_classes, in_channels=in_channels)
    if model in ("vit", "hybrid"):
        import vit
        return vit.build_model(num_classes, in_channels=in_channels,
                               variant=model, patch_size=patch)
    import text_models
    return text_models.build_model(model, num_classes)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """loader 전체에 대해 (정답, 예측, softmax 점수) 를 모아 반환한다."""
    model.eval()
    all_true, all_pred, all_score = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_true.append(yb.numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_score.append(probs.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_score),
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """한 에폭 학습하고 평균 손실을 반환한다."""
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


def fit(model, train_loader, val_loader, classes, class_weights, device,
        epochs: int, lr: float, patience: int):
    """학습 루프 + val Macro-F1 기준 조기 종료. best 가중치를 복원해 반환한다."""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_state, best_epoch, since_improved = -1.0, None, -1, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, _ = evaluate(model, val_loader, device)
        val_f1 = M.compute_metrics(y_true, y_pred, classes)["macro_f1"]
        dt = time.perf_counter() - t0
        print(f"  epoch {epoch:2d}/{epochs} | loss={train_loss:.4f} | val_macroF1={val_f1:.4f} | {dt:.1f}s")

        # 개선되면 best 갱신, 아니면 patience 카운트 증가.
        if val_f1 > best_f1:
            best_f1, best_epoch, since_improved = val_f1, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            since_improved += 1
            if since_improved >= patience:
                print(f"  [early stop] {patience}에폭 개선 없음 → 중단 (best epoch={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, best_epoch


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="RQ1 학습·평가 (제안 CNN + 텍스트 베이스라인)")
    parser.add_argument("--model", required=True, choices=sorted(IMAGE_MODELS | TEXT_MODELS))
    parser.add_argument("--track", default="payload_4class",
                        choices=["payload_4class", "payload_4class_csicnorm", "csic_binary",
                                 "ustc_flow_binary"])
    parser.add_argument("--text", default="raw", choices=["raw", "decoded"])
    parser.add_argument("--balance", action="store_true",
                        help="train 셋을 클래스 균형으로 언더샘플링(불균형 트랙 권장, RQ2 대비). "
                             "val/test 는 실제 분포 유지. 산출물 tag 에 '_bal' 접미사가 붙음")
    parser.add_argument("--side", type=int, default=48, help="이미지 한 변(이미지 모델 전용)")
    parser.add_argument("--channels", default="gray", choices=["gray", "rgb"],
                        help="이미지 채널(cnn 전용): gray=1채널, rgb=3채널(교수 요구). "
                             "rgb 는 먼저 build_image_dataset.py --channels rgb 로 빌드해야 함")
    parser.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy",
                        help="rgb 로드 시 R,G,B 채널 인코더 이름(콤마 구분). 빌드 때와 동일해야 함")
    parser.add_argument("--patch", default="8x8",
                        help="ViT 패치 크기 '높이x너비'(--model vit 전용). "
                             "'1x48' 은 행 패치 = 토큰 1개가 연속 48바이트(docs/08 §2). "
                             "hybrid 는 conv stem 이 대신하므로 무시됨")
    parser.add_argument("--max-len", type=int, default=48 * 48,
                        help="바이트 시퀀스 길이(텍스트 모델 전용). 기본=이미지 용량(48x48)과 동일")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=5, help="조기 종료 인내 에폭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="학습 샘플 수 제한(스모크용)")
    parser.add_argument("--smoke", action="store_true", help="빠른 동작 확인 모드(작게 실행)")
    # ── Phase 11 (RQ3) 방어 옵션 — docs/10 ────────────────────────────────
    parser.add_argument("--defense", default="none", choices=list(DEFENSE_MODES),
                        help="방어 방식: none=기준선, advtrain=의미보존 변형 치환 증강 학습, "
                             "norm=입력 정규화(디코딩; --text decoded 와 함께 써야 함)")
    parser.add_argument("--mutation-split", default="S0", choices=sorted(AUG.MUTATION_SPLITS),
                        help="증강에 쓸 변형 계열(advtrain 전용). S0=전 계열(seen, 낙관 상한), "
                             "SA=인코딩 계열 제외(held-out — 미지 변형 일반화 검증)")
    parser.add_argument("--aug-ratio", type=float, default=DEFAULT_AUG_RATIO,
                        help="공격 클래스에서 변형본으로 치환할 비율(0~1). 추가가 아니라 치환이라 "
                             "train 크기·클래스 비율은 불변")
    parser.add_argument("--aug-budget", type=int, default=DEFAULT_AUG_BUDGET,
                        help="한 샘플에 겹쳐 적용할 변형 개수 상한(학습용). 평가는 k=5 까지 간다")
    args = parser.parse_args()

    # 방어 옵션 정합성 검사 — 조용히 엉뚱한 실험을 돌리는 것보다 즉시 죽는 편이 낫다.
    if args.defense == "norm" and args.text != "decoded":
        parser.error("--defense norm 은 입력 정규화 방어이므로 --text decoded 와 함께 써야 합니다.")
    if args.defense == "advtrain" and args.track == "ustc_flow_binary":
        parser.error("--defense advtrain 은 페이로드 문자열 변형 기반이라 흐름 트랙에는 쓸 수 없습니다.")

    if args.smoke:
        # 코드가 안 깨지는지만 확인: 아주 작게, 1~2에폭.
        args.limit = args.limit or 500
        args.epochs = min(args.epochs, 2)

    set_seed(args.seed)
    device = get_device()
    print(f"=== 학습: model={args.model} track={args.track} text={args.text} device={device} ===")

    # rgb 채널 인코더 파싱(gray 모드에서는 무시됨)
    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))

    train_ds, val_ds, test_ds, classes, class_weights, in_channels, aug_info = build_datasets(
        args.model, args.track, args.text, args.side, args.max_len, args.limit,
        balance=args.balance, seed=args.seed, channels=args.channels, encoders=encoders,
        defense=args.defense, mutation_split=args.mutation_split,
        aug_ratio=args.aug_ratio, aug_budget=args.aug_budget,
    )
    if args.balance:
        print(f"  [balance] train 클래스 균형 언더샘플링 적용 → train={len(train_ds):,}")
    if args.defense == "advtrain":
        tr_info, va_info = aug_info["train"], aug_info["val"]
        print(f"  [defense] advtrain split={args.mutation_split} ratio={args.aug_ratio:g} "
              f"budget={args.aug_budget} → train 치환 {tr_info['n_replaced']:,}건 / "
              f"val 치환 {va_info['n_replaced']:,}건")
        print(f"    학습에 쓰는 변형({len(tr_info['techniques'])}종): {tr_info['techniques']}")
        print(f"    평가 전용 held-out({len(tr_info['held_out'])}종): {tr_info['held_out']}")
    elif args.defense == "norm":
        print("  [defense] norm — 입력 정규화(디코딩)본으로 학습. 학습 데이터 자체는 변형하지 않음")
    print(f"  train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,} "
          f"classes={classes}")
    print(f"  class_weights={np.round(class_weights, 3).tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, len(classes), in_channels=in_channels,
                        patch=args.patch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {n_params:,}")

    model, best_val_f1, best_epoch = fit(
        model, train_loader, val_loader, classes, class_weights, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )

    # 최종 test 평가 + 처리량 측정
    t0 = time.perf_counter()
    y_true, y_pred, y_score = evaluate(model, test_loader, device)
    infer_sec = time.perf_counter() - t0

    result = M.compute_metrics(y_true, y_pred, classes, y_score=y_score)
    result["throughput_samples_per_sec"] = float(len(y_true) / infer_sec) if infer_sec > 0 else None
    result["best_val_macro_f1"] = float(best_val_f1)
    result["best_epoch"] = int(best_epoch)
    result["n_params"] = int(n_params)
    result["model"] = args.model
    result["config"] = {
        "track": args.track, "text": args.text, "side": args.side,
        "max_len": args.max_len, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "device": str(device), "limit": args.limit,
        "balance": args.balance, "channels": args.channels, "in_channels": in_channels,
        "rgb_encoders": list(encoders) if (args.model in IMAGE_MODELS and args.channels == "rgb") else None,
        "patch": args.patch if args.model == "vit" else None,
    }
    # 방어 블록 — 파일명만으로는 복원되지 않는 정보(실제 치환 건수·계열 목록·조기종료 기준)를 남긴다.
    result["defense"] = {
        "mode": args.defense,
        "mutation_split": args.mutation_split if args.defense == "advtrain" else None,
        "aug_ratio": args.aug_ratio if args.defense == "advtrain" else None,
        "aug_budget": args.aug_budget if args.defense == "advtrain" else None,
        "seen_techniques": aug_info["train"]["techniques"] if aug_info["train"] else None,
        "held_out_techniques": aug_info["train"]["held_out"] if aug_info["train"] else None,
        "n_replaced_train": aug_info["train"]["n_replaced"] if aug_info["train"] else 0,
        "n_replaced_val": aug_info["val"]["n_replaced"] if aug_info["val"] else 0,
        # 조기 종료 기준을 arm 별 목적함수에 맞춘 근거는 docs/10 §7.3(ViT lr 함정과 동형).
        "early_stop_criterion": ("mixed_val_macro_f1" if args.defense == "advtrain"
                                 else "clean_val_macro_f1"),
    }

    if not args.smoke:
        # tag 규칙은 tagging.build_tag 단일 진실 소스를 쓴다(train/cross_validate/cascade/
        # run_evasion 4곳에 사본이 생기는 것을 막기 위함 — docs/10 §8).
        # 채널·패치·lr·방어·균형화가 모두 "실험을 가르는 축"이라 하나라도 빠지면 서로 덮어쓴다.
        is_image = args.model in IMAGE_MODELS
        tag = build_tag(
            args.track, args.model, args.text,
            channels=args.channels if is_image else "gray",
            encoders=encoders if is_image else None,
            patch=args.patch if args.model == "vit" else None,
            lr=args.lr, balance=args.balance,
            defense=args.defense, mutation_split=args.mutation_split, aug_ratio=args.aug_ratio,
        )
        M.save_report(result, RESULTS_DIR / f"{tag}.json")
        # 샘플 단위 예측 저장(모델 간 '탐지 불일치' 분석용 — detection_analysis.py 가 소비)
        M.save_predictions(y_true, y_pred, classes, RESULTS_DIR / f"pred_{tag}.npz", y_score=y_score)
        M.save_confusion_matrix(
            y_true, y_pred, classes, FIG_DIR / f"cm_{tag}.png",
            title=f"{args.model} ({args.track})",
        )
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
        print(f"  [저장] 지표/그림/체크포인트 → experiments/, docs/figures/models/")

    print("  " + M.format_summary(args.model, result))


if __name__ == "__main__":
    main()
