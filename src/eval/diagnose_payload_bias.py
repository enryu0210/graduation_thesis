"""
Phase 4 — payload_4class 성능이 왜 높게 나오는지(예: TF-IDF+LogReg AUC≈0.999) 진단하는 스크립트

배경:
    베이스라인이 clean test 에서 매우 높은 지표를 낸다. 이것이 (a) 데이터 누수 때문인지,
    (b) 과제 자체가 쉬워서인지, (c) 클래스 분포 편차(shortcut) 때문인지 구분해야
    논문 결과를 올바로 해석할 수 있다. 이 스크립트는 그 근거를 실측한다.

진단 3가지:
    1) 근접 중복 누수: train/test 를 정규화(소문자+공백축약)한 뒤 완전일치 비율.
       (Phase 3 의 exact dedup 은 raw 기준이라 이런 near-dup 은 못 잡음 → 별도 확인 필요)
    2) 클래스별 대표 페이로드: 표면 토큰만으로 얼마나 갈리는지 눈으로 확인.
    3) logreg 상위 char n-gram: 모델이 실제로 어떤 토큰에 의존하는지.

결론(2026-07 실행 기준, docs/04_models_rq1_design.md 7절에 기록):
    - 누수 아님(정규화 후 완전일치 0.3%).
    - 네 클래스의 표면 토큰이 거의 직교 → 선형 모델도 쉽게 분리.
    - 핵심 리스크: Normal 클래스가 '영어 산문(영화 리뷰류)'이라 공격(기호 범벅)과
      장르 자체가 달라, 모델이 '공격 여부'가 아니라 '문장이냐 기호냐'를 배우는 shortcut 위험.
      → 이 수치는 낙관적으로 부풀려진 것이며 실트래픽(CSIC)에는 그대로 통하지 않는다.

사용법:
    python src/eval/diagnose_payload_bias.py            # payload_4class raw
    (logreg 학습이 포함돼 수십 초 걸린다. torch 불필요.)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# data_text 로더 재사용
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from data_text import build_label_encoding, encode_labels_with, load_text_split  # noqa: E402


def normalize(text: str) -> str:
    """근접 중복 판정을 위한 최소 정규화: 소문자화 + 공백류 1칸 축약."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def report_leakage(tr_txt, te_txt, y_te, classes) -> None:
    """train/test 근접 중복(정규화 후 완전일치) 비율을 클래스별로 출력한다."""
    train_norm = set(map(normalize, tr_txt))
    te_norm = [normalize(t) for t in te_txt]
    leak = sum(1 for t in te_norm if t in train_norm)
    print(f"\n[1] 근접중복 누수: test {len(te_txt):,}건 중 정규화후 train 완전일치 "
          f"{leak:,}건 ({leak / len(te_txt) * 100:.1f}%)")
    for ci, c in enumerate(classes):
        idx = [i for i in range(len(te_txt)) if y_te[i] == ci]
        matched = sum(1 for i in idx if te_norm[i] in train_norm)
        print(f"     {c}: {matched / len(idx) * 100:.1f}% ({matched:,}/{len(idx):,})")


def report_examples(tr_txt, y_tr, classes, k: int = 3) -> None:
    """클래스별 대표 페이로드 k개를 출력한다(장르 차이를 눈으로 확인)."""
    print("\n[2] 클래스별 예시")
    tr_arr = np.array(tr_txt, dtype=object)
    for ci, c in enumerate(classes):
        print(f"  {c}:")
        for e in tr_arr[y_tr == ci][:k]:
            snippet = e[:80] + ("..." if len(e) > 80 else "")
            print(f"      {snippet}")


def report_top_features(tr_txt, y_tr, classes, top: int = 8) -> None:
    """logreg 가 클래스별로 의존하는 상위 char n-gram 을 출력한다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          max_features=20000, lowercase=False)
    X = vec.fit_transform(tr_txt)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    clf.fit(X, y_tr)
    feats = np.array(vec.get_feature_names_out())
    print("\n[3] 클래스별 logreg 상위 char n-gram")
    for ci, c in enumerate(classes):
        idx = np.argsort(clf.coef_[ci])[-top:][::-1]
        print(f"  {c}: {[repr(t) for t in feats[idx]]}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="payload_4class 고성능 원인 진단")
    parser.add_argument("--track", default="payload_4class",
                        choices=["payload_4class", "csic_binary"])
    parser.add_argument("--text", default="raw", choices=["raw", "decoded"])
    parser.add_argument("--skip-logreg", action="store_true",
                        help="상위 특징 분석(logreg 학습, 수십 초) 생략")
    args = parser.parse_args()

    tr_txt, tr_lab = load_text_split(args.track, "train", args.text)
    te_txt, te_lab = load_text_split(args.track, "test", args.text)
    y_tr, classes = build_label_encoding(tr_lab)
    y_te = encode_labels_with(te_lab, classes)
    print(f"=== 진단: track={args.track} text={args.text} classes={classes} ===")

    report_leakage(tr_txt, te_txt, y_te, classes)
    report_examples(tr_txt, y_tr, classes)
    if not args.skip_logreg:
        report_top_features(tr_txt, y_tr, classes)


if __name__ == "__main__":
    main()
