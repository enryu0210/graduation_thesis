"""
Phase 4/RQ1 보강 — 5-fold 교차검증 (docs/04 §6 "통계 검증" 항목 이행)

왜 필요한가:
    docs/04 §5 의 모델 순위는 단일 split(seed=42) 한 번의 결과다. 2026-07-18 재현 실험에서
    char-CNN 만 재실행 시 Accuracy 가 ±0.11pp 흔들렸고(cuDNN 비결정적 커널), 이 변동폭이
    RGB 채널 조합 간 차이(rb/cc/bd vs rb/ss/le 의 MCC 0.05pp)보다 **크다**.
    즉 "어느 조합/모델이 더 낫다"는 주장이 진짜 차이인지 노이즈인지 단일 split 으로는
    구분할 수 없다. → 같은 fold 위에서 설정들을 돌려 대응표본(paired) 비교를 가능하게 한다.

설계 결정:
    1) train/val/test 3분할을 하나의 풀(pool)로 합친 뒤 StratifiedKFold(k=5)로 재분할한다.
       fold 배정은 (라벨 배열, seed) 에만 의존하므로 **모든 설정이 정확히 같은 fold** 를 본다
       → paired t-test 의 전제(대응표본)가 성립한다. 라벨 지문을 저장해 정렬을 사후 검증한다.
    2) 각 fold 의 train 부분을 다시 90/10 으로 갈라 10% 를 조기종료용 val 로 쓴다.
       test fold 는 학습·조기종료에 일절 관여하지 않는다(누수 방지).
    3) 메모리: 풀 전체를 float32 로 올리면 RGB 기준 5GB 를 넘겨 OOM 위험이 있다.
       → 이미지는 uint8, 바이트 시퀀스는 int16 으로 들고 **배치 단위로만** 변환한다.

사용법:
    python src/eval/cross_validate.py --model cnn --channels rgb --rgb-encoders raw_byte,char_class,byte_delta
    python src/eval/cross_validate.py --model tfidf_rf
    → experiments/results/cv_{tag}.json (fold 별 전체 지표) 저장
    비교·유의성 판정은 cv_compare.py 가 담당한다(역할 분리).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# 이웃 모듈(src/models) import 경로 — train.py 와 동일한 관례를 따른다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

import metrics as M  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# 모델 종류별 입력 경로가 다르다(이미지 npz / 문자열 CSV / TF-IDF).
# ⚠️ train.py 의 IMAGE_MODELS 와 함께 갱신할 것(이미지 모델 추가 시 두 곳).
IMAGE_MODELS = {"cnn", "vit", "hybrid"}
TEXT_NN_MODELS = {"charcnn", "bilstm"}
TFIDF_MODELS = {"tfidf_logreg", "tfidf_rf"}
ALL_MODELS = IMAGE_MODELS | TEXT_NN_MODELS | TFIDF_MODELS

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# 데이터 풀 구성
# --------------------------------------------------------------------------
def label_fingerprint(y: np.ndarray) -> str:
    """라벨 배열의 해시. 설정마다 풀 순서가 같은지(=fold 가 대응되는지) 검증하는 용도."""
    return hashlib.sha256(np.asarray(y, dtype=np.int64).tobytes()).hexdigest()[:16]


def load_image_pool(track: str, text: str, side: int, channels: str,
                    encoders: tuple[str, ...] | None):
    """train/val/test npz 를 순서대로 이어붙여 (images uint8, y, classes) 풀을 만든다."""
    import data_image

    xs, ys, classes = [], [], None
    for split in SPLITS:
        x, y, cls = data_image.load_split(track, split, text, side, channels, encoders)
        xs.append(x)
        ys.append(y)
        # 클래스 순서가 split 마다 다르면 라벨 정수의 의미가 어긋난다 → 즉시 중단.
        if classes is not None and list(cls) != list(classes):
            raise ValueError(f"split 간 클래스 순서 불일치: {classes} vs {cls}")
        classes = cls
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), list(classes)


def load_text_pool(track: str, text: str):
    """train/val/test CSV 를 이어붙여 (texts list[str], y, classes) 풀을 만든다.

    라벨 정수화는 이미지 트랙과 동일한 sorted(unique) 규칙을 쓴다(data_text 위임).
    단 클래스 순서는 **풀 전체**로 정해야 split 별 결측 클래스 문제가 없다.
    """
    import pandas as pd
    from data_text import load_text_split

    texts: list[str] = []
    labels: list = []
    for split in SPLITS:
        t, lab = load_text_split(track, split, text)
        texts.extend(t)
        labels.append(lab)
    labels_all = pd.concat(labels, ignore_index=True)
    classes = sorted(labels_all.unique())
    code = {name: i for i, name in enumerate(classes)}
    y = labels_all.map(code).to_numpy(dtype=np.int64)
    return texts, y, classes


# --------------------------------------------------------------------------
# 메모리 절약형 Dataset (uint8/int16 보관 → 배치에서 변환)
# --------------------------------------------------------------------------
def make_lazy_image_dataset(images: np.ndarray, labels: np.ndarray):
    """(N,H,W) 또는 (N,H,W,3) uint8 을 그대로 들고, 꺼낼 때만 float(0~1)로 바꾼다.

    train.py 의 make_torch_dataset 은 풀 전체를 float32 로 복제하는데(RGB 20만 장 ≈ 5.5GB),
    CV 는 fold 마다 데이터셋을 다시 만들어 OOM 위험이 크다. 값 자체는 동일하고
    메모리 표현만 다르므로 학습 결과는 바뀌지 않는다.
    """
    import torch
    from torch.utils.data import Dataset

    if images.ndim == 3:                       # gray (N,H,W) → (N,1,H,W)
        x = torch.from_numpy(images).unsqueeze(1)
    elif images.ndim == 4 and images.shape[-1] == 3:   # rgb HWC → CHW
        x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"지원하지 않는 이미지 배열 형태입니다: {images.shape}")
    y = torch.from_numpy(labels.astype(np.int64))

    class LazyImages(Dataset):
        def __len__(self):
            return x.shape[0]

        def __getitem__(self, i):
            return x[i].float().div(255.0), y[i]

    return LazyImages()


def make_lazy_seq_dataset(seqs: np.ndarray, labels: np.ndarray):
    """(N,L) int16 바이트 시퀀스를 들고, 꺼낼 때만 int64(Embedding 요구 dtype)로 바꾼다.

    int64 로 풀 전체를 들면 20만×2304×8B ≈ 3.7GB → int16 이면 1/4 로 줄어든다.
    값 범위는 0~256 이라 int16 으로 무손실이다.
    """
    import torch
    from torch.utils.data import Dataset

    x = torch.from_numpy(seqs)
    y = torch.from_numpy(labels.astype(np.int64))

    class LazySeqs(Dataset):
        def __len__(self):
            return x.shape[0]

        def __getitem__(self, i):
            return x[i].long(), y[i]

    return LazySeqs()


def encode_byte_matrix_int16(texts: list[str], max_len: int) -> np.ndarray:
    """문자열 리스트 → (N,max_len) int16 바이트 시퀀스(패딩 0, 바이트값+1).

    data_text.encode_byte_sequence 와 **동일 규칙**이되 dtype 만 int16 이다
    (풀 전체를 int64 로 들면 메모리 초과 — 위 make_lazy_seq_dataset 주석 참조).
    """
    matrix = np.zeros((len(texts), max_len), dtype=np.int16)
    for i, text in enumerate(texts):
        raw = text.encode("utf-8", errors="replace")[:max_len]
        if raw:
            matrix[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) + 1
    return matrix


# --------------------------------------------------------------------------
# fold 단위 학습·평가
# --------------------------------------------------------------------------
def inner_train_val_split(y_trainval: np.ndarray, val_ratio: float, seed: int):
    """fold 의 train 부분을 (train, val) 로 층화 분할한다 — 조기종료 전용 val 확보."""
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_rel, va_rel = next(splitter.split(np.zeros(len(y_trainval)), y_trainval))
    return tr_rel, va_rel


def run_fold_torch(model_name: str, pool_x, y, classes, tr_idx, va_idx, te_idx,
                   args, is_image: bool) -> dict:
    """torch 모델(cnn/charcnn/bilstm) 한 fold 학습 → test fold 지표 반환."""
    import torch
    from torch.utils.data import DataLoader
    import data_image
    import train as T

    make_ds = make_lazy_image_dataset if is_image else make_lazy_seq_dataset
    train_ds = make_ds(pool_x[tr_idx], y[tr_idx])
    val_ds = make_ds(pool_x[va_idx], y[va_idx])
    test_ds = make_ds(pool_x[te_idx], y[te_idx])

    class_weights = data_image.compute_class_weights(y[tr_idx], len(classes))
    device = T.get_device()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 이미지 채널 수는 배열 형태에서 추론(train.py 와 동일 규칙).
    in_channels = 3 if (is_image and pool_x.ndim == 4 and pool_x.shape[-1] == 3) else 1
    model = T.build_model(model_name, len(classes), in_channels=in_channels,
                          patch=args.patch).to(device)

    model, best_val_f1, best_epoch = T.fit(
        model, train_loader, val_loader, classes, class_weights, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )

    t0 = time.perf_counter()
    y_true, y_pred, y_score = T.evaluate(model, test_loader, device)
    infer_sec = time.perf_counter() - t0

    result = M.compute_metrics(y_true, y_pred, classes, y_score=y_score)
    result["throughput_samples_per_sec"] = float(len(y_true) / infer_sec) if infer_sec > 0 else None
    result["best_val_macro_f1"] = float(best_val_f1)
    result["best_epoch"] = int(best_epoch)

    # fold 마다 GPU 메모리를 반납하지 않으면 5폴드 누적으로 OOM 이 난다.
    del model, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_fold_tfidf(model_name: str, texts: list[str], y, classes, tr_idx, te_idx,
                   args) -> dict:
    """TF-IDF + 전통 ML 한 fold 학습 → test fold 지표 반환(조기종료 없음 → val 불필요)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from baseline_tfidf import build_classifier

    clf_name = model_name.replace("tfidf_", "")
    train_texts = [texts[i] for i in tr_idx]
    test_texts = [texts[i] for i in te_idx]

    # 벡터라이저는 **train fold 에만** 적합시킨다(test 어휘 누수 방지).
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=args.max_features, lowercase=False,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = build_classifier(clf_name, len(classes))
    t0 = time.perf_counter()
    clf.fit(X_train, y[tr_idx])
    fit_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_sec = time.perf_counter() - t0
    y_score = clf.predict_proba(X_test)

    result = M.compute_metrics(y[te_idx], y_pred, classes, y_score=y_score)
    result["throughput_samples_per_sec"] = float(len(te_idx) / infer_sec) if infer_sec > 0 else None
    result["fit_seconds"] = float(fit_sec)
    return result


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="RQ1 5-fold 교차검증 (설정 1개 실행)")
    parser.add_argument("--model", required=True, choices=sorted(ALL_MODELS))
    parser.add_argument("--track", default="payload_4class",
                        choices=["payload_4class", "payload_4class_csicnorm", "csic_binary",
                                 "ustc_flow_binary"])
    parser.add_argument("--text", default="raw", choices=["raw", "decoded"])
    parser.add_argument("--side", type=int, default=48)
    parser.add_argument("--channels", default="gray", choices=["gray", "rgb"])
    parser.add_argument("--rgb-encoders", default="raw_byte,char_class,local_entropy")
    parser.add_argument("--patch", default="8x8", help="ViT 패치 '높이x너비'(--model vit 전용)")
    parser.add_argument("--max-len", type=int, default=48 * 48)
    parser.add_argument("--max-features", type=int, default=20000, help="TF-IDF 특징 수")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="fold train 중 조기종료용 val 비율")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sklearn.model_selection import StratifiedKFold

    encoders = tuple(name.strip() for name in args.rgb_encoders.split(","))
    is_image = args.model in IMAGE_MODELS
    is_tfidf = args.model in TFIDF_MODELS

    print(f"=== 5-fold CV: model={args.model} track={args.track} "
          f"channels={args.channels if is_image else '-'} ===")

    # 1) 풀 구성
    if is_image:
        pool_x, y, classes = load_image_pool(args.track, args.text, args.side,
                                             args.channels, encoders)
        pool_texts = None
    else:
        pool_texts, y, classes = load_text_pool(args.track, args.text)
        pool_x = None
        if not is_tfidf:
            print("  바이트 시퀀스 인코딩 중...")
            pool_x = encode_byte_matrix_int16(pool_texts, args.max_len)

    fp = label_fingerprint(y)
    print(f"  풀 크기={len(y):,} classes={classes} 라벨지문={fp}")

    # 2) fold 배정 — (y, seed) 에만 의존 → 모든 설정이 같은 fold 를 본다.
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y))

    import train as T
    fold_results = []
    for k, (trval_idx, te_idx) in enumerate(folds, start=1):
        # 시드는 fold 마다 다르게(재현 가능하게) 줘서 초기값 우연을 분산시킨다.
        T.set_seed(args.seed + k)
        t0 = time.perf_counter()
        print(f"\n--- fold {k}/{args.folds} (train+val={len(trval_idx):,} test={len(te_idx):,}) ---")

        if is_tfidf:
            res = run_fold_tfidf(args.model, pool_texts, y, classes, trval_idx, te_idx, args)
        else:
            tr_rel, va_rel = inner_train_val_split(y[trval_idx], args.val_ratio, args.seed + k)
            res = run_fold_torch(args.model, pool_x, y, classes,
                                 trval_idx[tr_rel], trval_idx[va_rel], te_idx,
                                 args, is_image=is_image)
        res["fold"] = k
        res["elapsed_sec"] = round(time.perf_counter() - t0, 1)
        fold_results.append(res)
        print(f"  fold {k}: " + M.format_summary(args.model, res))

    # 3) fold 간 요약(평균±표준편차) — 유의성 판정은 cv_compare.py 담당
    keys = ["accuracy", "macro_f1", "mcc", "pr_auc_macro"]
    summary = {}
    for key in keys:
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                            "per_fold": [float(v) for v in vals]}
    # benign-evasion 은 attack_focused 안에 있어 따로 추출한다(보안 헤드라인 지표).
    ev = [r["attack_focused"]["benign_evasion_rate"] for r in fold_results
          if "attack_focused" in r]
    if ev:
        summary["benign_evasion_rate"] = {"mean": float(np.mean(ev)),
                                          "std": float(np.std(ev, ddof=1)),
                                          "per_fold": [float(v) for v in ev]}

    # tag 규칙은 tagging.build_tag 단일 진실 소스를 쓴다(train.py 와 자동으로 일치).
    # ⚠️ Phase 11 방어 arm 의 CV 는 아직 미지원이다 — cross_validate 는 npz 풀에서 fold 를
    #    나누는데, 증강은 원문 텍스트가 필요해 풀 구성 자체가 달라진다(docs/10 §7.4, §10).
    #    지원 전까지는 방어 arm 판정을 단일 split 효과 크기로만 한다(docs/10 §6).
    from tagging import build_tag
    tag = build_tag(
        args.track, args.model, args.text,
        channels=args.channels if is_image else "gray",
        encoders=encoders if is_image else None,
        patch=args.patch if args.model == "vit" else None,
        lr=args.lr,
    )

    payload = {
        "model": args.model,
        "tag": tag,
        "folds": args.folds,
        "seed": args.seed,
        "label_fingerprint": fp,   # 설정 간 fold 정렬 검증용
        "classes": classes,
        "n_pool": int(len(y)),
        "config": {"track": args.track, "text": args.text, "side": args.side,
                   "channels": args.channels if is_image else None,
                   "rgb_encoders": list(encoders) if (is_image and args.channels == "rgb") else None,
                   "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                   "val_ratio": args.val_ratio},
        "summary": summary,
        "fold_results": fold_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"cv_{tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {tag} 완료 → {out.name} ===")
    for key, s in summary.items():
        print(f"  {key:22} {s['mean']:.4f} ± {s['std']:.4f}")


if __name__ == "__main__":
    main()
