"""
Phase 4.5 — 모델 간 '탐지 불일치' 분석 (RQ1 재해석: 무엇을 잡고 무엇을 놓치는가)

목적:
    Macro-F1 같은 집계 지표는 "이미지 CNN 이 텍스트 베이스라인이 *못 잡던 공격*을 잡는가"라는
    질문에 답하지 못한다(전체 정확도가 비슷해도 '잡는 대상'이 다를 수 있다). 이 스크립트는
    각 모델이 남긴 샘플 단위 예측(pred_*.npz)을 test CSV(원문 페이로드)와 행 인덱스로 join 해,
      1) 기준 모델(예: 이미지 CNN) vs 베이스라인의 '탐지 상보성(complementarity)' 표
      2) 기준 모델만 탐지에 성공한 실제 페이로드 목록(케이스 스터디 CSV)
    를 만든다.

'탐지 성공'의 정의(보안 관점, metrics.attack_focused 와 일치):
    진짜 공격 샘플에 대해 예측이 Normal 이 아니면(= 어떤 공격 클래스로든 분류) '탐지', Normal 로
    새면 '미탐(benign-evasion)'. WAF 관점에서 실제 위험은 '공격을 정상으로 흘려보내는 것'이므로
    이 정의가 논문 주장("남들이 놓친 공격을 잡는다")과 정확히 대응한다.

의존성: pandas / numpy 만. torch 불필요(예측은 이미 npz 로 저장돼 있음).

사용 예:
    # tfidf 두 모델 비교(로컬 CPU 로 바로 재현 가능)
    python src/eval/detection_analysis.py --track payload_4class_csicnorm \
        --ref tfidf_rf --baselines tfidf_logreg

    # 본 비교(GPU 학습 후): 이미지 CNN 이 텍스트 모델들이 놓친 걸 잡는가
    python src/eval/detection_analysis.py --track payload_4class_csicnorm --bal \
        --ref cnn --baselines tfidf_logreg,tfidf_rf,charcnn,bilstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CASES_DIR = RESULTS_DIR / "detection_cases"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 정상 라벨 판별(metrics._find_normal_index 와 동일 규칙: 관대한 부분일치)
NORMAL_KEYS = ("normal", "benign", "valid")


def resolve_pred_path(model: str, track: str, text: str, bal: bool) -> Path:
    """모델 짧은 이름을 학습 스크립트의 태그 규칙에 맞는 pred_*.npz 경로로 변환한다.

    - tfidf_* : baseline_tfidf.py 규칙 → pred_{track}_tfidf_{clf}_{text}.npz (bal 접미사 없음)
    - 그 외(cnn/charcnn/bilstm) : train.py 규칙 → pred_{track}_{model}_{text}[_bal].npz
    """
    if model.startswith("tfidf_"):
        # tfidf 는 class_weight='balanced' 로 자체 보정하므로 _bal 접미사를 붙이지 않는다.
        return RESULTS_DIR / f"pred_{track}_{model}_{text}.npz"
    suffix = "_bal" if bal else ""
    return RESULTS_DIR / f"pred_{track}_{model}_{text}{suffix}.npz"


def load_predictions(path: Path) -> dict:
    """pred_*.npz 를 로드해 {y_true, y_pred, class_names, normal_idx} 로 반환한다."""
    if not path.exists():
        raise FileNotFoundError(f"예측 파일이 없습니다: {path}\n"
                                f"  → 해당 모델을 먼저 학습/평가해 pred_*.npz 를 생성하세요.")
    data = np.load(path, allow_pickle=True)
    class_names = [str(c) for c in data["class_names"]]
    normal_idx = _find_normal_index(class_names)
    if normal_idx is None:
        raise ValueError(f"{path.name}: 정상 클래스(Normal/benign)를 찾지 못했습니다. "
                         f"이 분석은 '공격→Normal 누출'을 기준으로 하므로 정상 클래스가 필요합니다.")
    return {
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "class_names": class_names,
        "normal_idx": normal_idx,
    }


def _find_normal_index(class_names: list[str]) -> int | None:
    for i, name in enumerate(class_names):
        if any(key in name.lower() for key in NORMAL_KEYS):
            return i
    return None


def detected_mask(y_pred: np.ndarray, normal_idx: int) -> np.ndarray:
    """'탐지 성공' 마스크: 예측이 Normal 이 아니면 True(= 공격으로 잡음)."""
    return np.asarray(y_pred) != normal_idx


def compare(ref: dict, base: dict, texts_raw: list[str], labels: pd.Series,
            ref_name: str, base_name: str) -> dict:
    """기준 모델 vs 베이스라인의 탐지 상보성을 계산하고 케이스 CSV 를 저장한다."""
    # 두 예측이 같은 test 셋(같은 행 순서)인지 방어적으로 확인한다.
    if len(ref["y_true"]) != len(base["y_true"]):
        raise ValueError(f"표본 수 불일치: {ref_name}={len(ref['y_true'])} vs "
                         f"{base_name}={len(base['y_true'])} — 같은 track/split 인지 확인하세요.")
    if not np.array_equal(ref["y_true"], base["y_true"]):
        raise ValueError("두 예측의 y_true 가 다릅니다 — 같은 test 셋이 아닙니다(정렬/트랙 확인).")

    y_true = ref["y_true"]
    n_idx = ref["normal_idx"]
    is_attack = y_true != n_idx  # 진짜 공격 샘플만 분석 대상(정상→정상은 탐지 이슈 아님)

    ref_det = detected_mask(ref["y_pred"], n_idx)
    base_det = detected_mask(base["y_pred"], n_idx)

    both = is_attack & ref_det & base_det
    ref_only = is_attack & ref_det & ~base_det      # 핵심: 기준만 잡음(남들이 놓친 공격)
    base_only = is_attack & ~ref_det & base_det
    neither = is_attack & ~ref_det & ~base_det

    n_attack = int(is_attack.sum())
    summary = {
        "ref": ref_name, "baseline": base_name, "n_attacks": n_attack,
        "both_detected": int(both.sum()),
        "ref_only_detected": int(ref_only.sum()),      # ← 논문 주장의 직접 근거
        "baseline_only_detected": int(base_only.sum()),
        "neither_detected": int(neither.sum()),
        "ref_detection_recall": float(ref_det[is_attack].mean()) if n_attack else None,
        "baseline_detection_recall": float(base_det[is_attack].mean()) if n_attack else None,
    }

    # 케이스 스터디 저장: '기준만 탐지'한 실제 페이로드(=베이스라인이 Normal 로 흘린 공격)
    _export_cases(ref_only, texts_raw, labels, ref, base,
                  CASES_DIR / f"{ref_name}_vs_{base_name}_refONLY.csv")
    # 대칭 근거(과장 방지): 베이스라인만 탐지한 것도 함께 남긴다.
    _export_cases(base_only, texts_raw, labels, ref, base,
                  CASES_DIR / f"{ref_name}_vs_{base_name}_baseONLY.csv")
    return summary


def _export_cases(mask: np.ndarray, texts_raw: list[str], labels: pd.Series,
                  ref: dict, base: dict, out_path: Path) -> None:
    """마스크에 해당하는 샘플의 (원문 페이로드, 진짜 라벨, 양쪽 예측)을 CSV 로 저장한다."""
    idxs = np.where(mask)[0]
    cnames = ref["class_names"]
    rows = [{
        "idx": int(i),
        "label": labels.iloc[i],
        "ref_pred": cnames[int(ref["y_pred"][i])],
        "base_pred": cnames[int(base["y_pred"][i])],
        # 원문 페이로드는 개행/쉼표가 있어 CSV 인용이 필요 → pandas 가 처리
        "text_raw": texts_raw[i],
    } for i in idxs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["idx", "label", "ref_pred", "base_pred", "text_raw"]).to_csv(
        out_path, index=False, encoding="utf-8"
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description="모델 간 탐지 불일치 분석(RQ1 재해석)")
    p.add_argument("--track", required=True,
                   choices=["payload_4class", "payload_4class_csicnorm", "csic_binary"])
    p.add_argument("--text", default="raw", choices=["raw", "decoded"])
    p.add_argument("--bal", action="store_true", help="torch 모델의 _bal(균형화) 예측을 사용")
    p.add_argument("--ref", required=True, help="기준 모델(예: cnn) — '남들이 못 잡던 걸 잡는' 주체")
    p.add_argument("--baselines", required=True,
                   help="비교 베이스라인들(쉼표 구분, 예: tfidf_logreg,tfidf_rf,charcnn,bilstm)")
    args = p.parse_args()

    # test CSV(원문 페이로드)를 한 번만 로드해 모든 비교에서 공유한다.
    csv_path = PROCESSED_DIR / f"{args.track}_test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"전처리 CSV 없음: {csv_path} (preprocess.py 먼저 실행)")
    df = pd.read_csv(csv_path, encoding="utf-8")
    texts_raw = df["text_raw"].fillna("").astype(str).tolist()

    ref = load_predictions(resolve_pred_path(args.ref, args.track, args.text, args.bal))
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    print(f"=== 탐지 불일치 분석: track={args.track} ref={args.ref} ===")
    print(f"{'baseline':<16} {'공격수':>7} {'둘다':>7} {'ref만':>7} {'base만':>7} {'둘다놓침':>8} "
          f"{'ref재현':>8} {'base재현':>8}")
    summaries = []
    for b in baselines:
        try:
            base = load_predictions(resolve_pred_path(b, args.track, args.text, args.bal))
            s = compare(ref, base, texts_raw, df["label"], args.ref, b)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [건너뜀] {b}: {e}")
            continue
        summaries.append(s)
        print(f"{b:<16} {s['n_attacks']:>7,} {s['both_detected']:>7,} "
              f"{s['ref_only_detected']:>7,} {s['baseline_only_detected']:>7,} "
              f"{s['neither_detected']:>8,} "
              f"{s['ref_detection_recall']:>8.4f} {s['baseline_detection_recall']:>8.4f}")

    if summaries:
        import json
        out = RESULTS_DIR / f"detection_complementarity_{args.track}_{args.ref}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] 상보성 요약 → {out.relative_to(PROJECT_ROOT)}")
        print(f"[저장] 케이스 CSV → {CASES_DIR.relative_to(PROJECT_ROOT)}/ "
              f"({args.ref}_vs_*_refONLY.csv = 기준만 잡은 실제 페이로드)")
    else:
        print("\n비교할 예측 파일을 찾지 못했습니다. 먼저 각 모델을 학습/평가하세요.")


if __name__ == "__main__":
    main()
