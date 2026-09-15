"""감사 원자료를 보존하고 트랙과의 정확한 중복을 드러내기 위한 평가 세트다."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# 직접 실행해도 기존 데이터 정의와 지표 구현을 그대로 사용한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.srbh_track import AUDIT_CSV, RAW_DIR, FILENAME, compose_fields
from src.eval.metrics import wilson_ci


def build_e1a_set(raw: pd.DataFrame, flags: pd.DataFrame,
                  track: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """중복 행도 감사 증거이므로 삭제하지 않고 누수 여부만 별도로 표시한다."""
    required = {"row_id", "audit_classes", "patterns", "fields"}
    if not required <= set(flags.columns) or not raw.index.is_unique:
        raise ValueError("감사 컬럼과 중복 없는 원본 인덱스가 필요합니다.")
    ids = flags.row_id.astype(str)
    if not ids.str.fullmatch(r"[0-9]+").all():
        raise ValueError("감사 row_id는 0 이상의 정수여야 합니다.")
    ids = ids.map(int)
    if not ids.isin(raw.index).all():
        raise ValueError("감사 row_id가 원본에 없습니다.")
    selected = raw.loc[ids]
    result = flags[["row_id", "audit_classes", "patterns", "fields"]].reset_index(drop=True).copy()
    result["row_id"] = ids.to_numpy()
    result["text_f2"] = compose_fields(selected, "F2")
    result["text_f3"] = compose_fields(selected, "F3")
    result["outside_f2"] = result.fields.map(
        lambda value: not {"request_http_request", "request_body"}.intersection(str(value).split(";")))
    result["overlap_split"] = ""
    overlap = {}
    for split, frame in track.items():
        mask = result.text_f2.isin(frame.text_raw)
        overlap[split] = {"unique_texts": int(result.loc[mask, "text_f2"].nunique()),
                          "rows": int(mask.sum())}
        # 여러 split에 걸친 중복도 숨기지 않도록 모두 기록한다.
        result.loc[mask, "overlap_split"] += split + ";"
    result["overlap_split"] = result.overlap_split.str.rstrip(";")
    stats = {"n_rows": len(result), "n_unique_text_f2": int(result.text_f2.nunique()),
             "n_unique_text_f3": int(result.text_f3.nunique()),
             "n_outside_f2_rows": int(result.outside_f2.sum()), "overlap_with_track": overlap}
    return result[["row_id", "text_f2", "text_f3", "audit_classes", "patterns", "fields",
                   "outside_f2", "overlap_split"]], stats


def detection_report(pred_labels: Sequence[str], normal_label: str = "Normal",
                     confidence=0.95) -> dict:
    """공격 세트의 Normal 예측을 회피로 세고 공통 Wilson 구간으로 보고한다."""
    n = len(pred_labels)
    detected = sum(label != normal_label for label in pred_labels)
    return {"n": n, "detected": int(detected),
            "detection_rate": wilson_ci(int(detected), n, confidence),
            "benign_evasion": wilson_ci(int(n - detected), n, confidence)}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """두 분석 CLI가 동일한 문자열 로딩 규칙으로 F2 바이트를 보존하게 한다."""
    paths = {split: PROJECT_ROOT / f"data/processed/srbh_4class_{split}.csv"
             for split in ("train", "val", "test")}
    if not AUDIT_CSV.is_file():
        raise FileNotFoundError("감사 CSV가 없습니다. python src/data/audit_srbh_labels.py 를 실행하세요.")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "트랙 CSV가 없습니다. 재생성 명령: python -c \"import sys; "
            "sys.path.insert(0,'src/data'); import preprocess; preprocess.process_track('srbh_4class')\""
            " (기존 실험 지문 확인 후 별도 실행)")
    options = dict(dtype=str, keep_default_na=False, encoding="utf-8")
    raw = pd.read_csv(RAW_DIR / FILENAME, **options)
    return raw, pd.read_csv(AUDIT_CSV, **options), {
        split: pd.read_csv(path, **options) for split, path in paths.items()}


def main() -> int:
    """출력 경로를 평가 전용으로 제한해 기존 트랙을 덮어쓰지 않는다."""
    try:
        frame, stats = build_e1a_set(*load_inputs())
        csv_path = PROJECT_ROOT / "data/processed/srbh_e1a.csv"
        json_path = PROJECT_ROOT / "experiments/results/srbh_e1_sets.json"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        summary = json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False)
        json_path.write_text(summary + "\n", encoding="utf-8")
        print(summary)
    except (OSError, ValueError, KeyError) as error:
        print(f"E1-a 평가 세트 생성 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
