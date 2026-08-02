"""
Phase 4 — 베이스라인 ①: TF-IDF (문자 n-gram) + 전통 ML (RQ1 비교군)

목적:
    설계 6.1(RQ1)의 "전통 ML 베이스라인: TF-IDF/n-gram + RandomForest 등"을 구현한다.
    이미지 기반 RGB CNN 과 "같은 test 셋 / 같은 지표(metrics.py)"로 비교해,
    이미지화가 텍스트 특징공학 대비 실익이 있는지 판단할 근거를 만든다.

왜 문자(char) n-gram 인가:
    웹 공격 페이로드는 `<`, `'`, `--`, `;` 같은 특수문자 토큰과 짧은 조각이 핵심 신호다.
    단어 단위보다 문자 n-gram 이 이런 구두점 패턴을 잘 잡고, 이미지 트랙이 쓰는
    "바이트 관점"과도 철학이 맞아 비교가 공정하다.

이 스크립트는 torch 불필요 — scikit-learn 만으로 이 노트북에서 바로 실행된다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 같은 폴더/이웃 폴더 모듈 import (스크립트 직접 실행 대비)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from data_text import build_label_encoding, encode_labels_with, load_text_split  # noqa: E402
import metrics as M  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "models"


def build_classifier(name: str, n_classes: int):
    """이름으로 분류기를 생성한다. class_weight='balanced' 로 불균형을 보정한다."""
    if name == "logreg":
        # saga: 대규모 희소 데이터에 적합, 멀티클래스 지원. 반복수는 넉넉히.
        return LogisticRegression(
            max_iter=1000, n_jobs=-1, class_weight="balanced", C=1.0,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42,
        )
    raise ValueError(f"알 수 없는 분류기: {name} (logreg | rf)")


def run(track: str, text: str, clf_name: str, max_features: int) -> dict:
    """TF-IDF + 분류기를 학습하고 test 셋 지표를 계산·저장한다. 반환: 지표 dict."""
    # 1) 데이터 로드 (train 으로 클래스 순서를 확정하고 test 를 같은 매핑으로 인코딩)
    train_texts, train_labels_str = load_text_split(track, "train", text)
    test_texts, test_labels_str = load_text_split(track, "test", text)
    y_train, classes = build_label_encoding(train_labels_str)
    y_test = encode_labels_with(test_labels_str, classes)

    print(f"  train={len(train_texts):,} test={len(test_texts):,} classes={classes}")

    # 2) TF-IDF (문자 n-gram 2~4). max_features 로 차원을 제한해 메모리/속도 관리.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=max_features, lowercase=False,
    )
    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"  TF-IDF: {X_train.shape[1]:,} features ({time.perf_counter() - t0:.1f}s)")

    # 3) 학습
    clf = build_classifier(clf_name, len(classes))
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0
    print(f"  fit: {fit_sec:.1f}s")

    # 4) 예측 + 처리량(throughput) 측정
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_sec = time.perf_counter() - t0
    y_score = clf.predict_proba(X_test)  # logreg/rf 모두 확률 제공 → ROC-AUC 계산 가능

    # 5) 지표 계산·저장 (metrics.py 로 일원화)
    result = M.compute_metrics(y_test, y_pred, classes, y_score=y_score)
    result["throughput_samples_per_sec"] = float(len(y_test) / infer_sec) if infer_sec > 0 else None
    result["fit_seconds"] = float(fit_sec)
    result["model"] = f"tfidf_{clf_name}"
    result["config"] = {"track": track, "text": text, "max_features": max_features}

    tag = f"{track}_tfidf_{clf_name}_{text}"
    M.save_report(result, RESULTS_DIR / f"{tag}.json")
    # 샘플 단위 예측 저장(모델 간 '탐지 불일치' 분석용 — detection_analysis.py 가 소비)
    M.save_predictions(y_test, y_pred, classes, RESULTS_DIR / f"pred_{tag}.npz", y_score=y_score)
    M.save_confusion_matrix(
        y_test, y_pred, classes, FIG_DIR / f"cm_{tag}.png",
        title=f"TF-IDF+{clf_name} ({track})",
    )
    print("  " + M.format_summary(result["model"], result))
    return result


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="RQ1 베이스라인: TF-IDF + 전통 ML")
    parser.add_argument("--track", default="payload_4class",
                        choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"])
    parser.add_argument("--text", default="raw", choices=["raw", "decoded"])
    parser.add_argument("--clf", default="logreg", choices=["logreg", "rf"],
                        help="분류기: logreg(빠름·강력) | rf(RandomForest)")
    parser.add_argument("--max-features", type=int, default=20000,
                        help="TF-IDF 최대 특징 수(차원 제한)")
    args = parser.parse_args()

    print(f"=== TF-IDF 베이스라인: track={args.track} text={args.text} clf={args.clf} ===")
    run(args.track, args.text, args.clf, args.max_features)


if __name__ == "__main__":
    main()
