"""SR-BH의 감사 제외와 라벨 충돌 제거를 공통 분할보다 먼저 적용한다."""

from __future__ import annotations

import pandas as pd

if __package__:
    from .download_srbh import FILENAME, PROJECT_ROOT, RAW_DIR
    from .profile_srbh import TEXT_FIELDS, detect_label_columns
else:
    from download_srbh import FILENAME, PROJECT_ROOT, RAW_DIR
    from profile_srbh import TEXT_FIELDS, detect_label_columns

AUDIT_CSV = PROJECT_ROOT / "data/processed/srbh_audit_flags.csv"
CLASS_CODES = {
    "Normal": ("000",), "SQLInjection": ("66",),
    "CodeInjection": ("242",), "CommandInjection": ("88", "248"),
}
OUTPUT_COLUMNS = ["text_raw", "label", "row_id", *TEXT_FIELDS]


def _counts(frame: pd.DataFrame) -> dict:
    """제거로 클래스가 사라져도 0건을 명시해 단계 간 대조를 가능하게 한다."""
    return {"total": len(frame), "by_class": {
        label: int(frame["label"].eq(label).sum()) for label in CLASS_CODES
    }}


def build_srbh_frame(raw: pd.DataFrame, flagged_row_ids: set[int]) -> tuple[pd.DataFrame, dict]:
    """원본 인덱스를 감사 행 번호로 보존하며 파일 접근 없이 트랙을 구성한다."""
    labels = detect_label_columns(raw)
    by_code = {name.split(" - ", 1)[0]: name for name in labels}
    required_codes = {code for codes in CLASS_CODES.values() for code in codes}
    if not required_codes <= by_code.keys():
        raise ValueError("매핑에 필요한 라벨 코드가 없습니다.")
    if not set(TEXT_FIELDS) <= set(raw.columns):
        raise ValueError("트랙에 필요한 텍스트 컬럼이 없습니다.")
    if not raw.index.is_unique or any(not isinstance(i, int) or i < 0 for i in raw.index):
        raise ValueError("원본 행 번호는 중복 없는 0 이상의 정수여야 합니다.")
    if not raw[labels].isin(["0", "1"]).all().all():
        raise ValueError("라벨에 문자열 0 또는 1 이외의 값이 있습니다.")
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in flagged_row_ids):
        raise ValueError("감사 row_id는 0 이상의 정수여야 합니다.")

    # 88과 248이 함께 켜져도 CommandInjection 한 클래스로 판단한다.
    targets = pd.DataFrame({
        label: raw[[by_code[code] for code in codes]].eq("1").any(axis=1)
        for label, codes in CLASS_CODES.items()
    }, index=raw.index)
    cardinality = targets.sum(axis=1)
    single_class = cardinality.eq(1)
    frame = raw.loc[single_class, list(TEXT_FIELDS)].copy()
    frame["row_id"] = frame.index
    frame["label"] = targets.loc[single_class].idxmax(axis=1)
    stats = {
        "n_raw": len(raw), "ambiguous": int(cardinality.ge(2).sum()),
        "non_target": int(cardinality.eq(0).sum()), "after_mapping": _counts(frame),
    }
    # 잘못된 감사 파일이나 다른 원본의 행 번호를 조용히 무시하면 오염이 남는다.
    normal_ids = set(frame.index[frame["label"].eq("Normal")])
    invalid_ids = flagged_row_ids - normal_ids
    if invalid_ids:
        raise ValueError(f"감사 row_id가 매핑 후 Normal 행을 가리키지 않습니다: {sorted(invalid_ids)[:10]}")
    flagged = frame.index.isin(flagged_row_ids)
    stats["e1a_removed"] = int(flagged.sum())
    frame = frame.loc[~flagged].copy()
    # 빈 합성 입력에서도 pandas의 숫자형 추론 때문에 문자열 결합이 실패하지 않게 한다.
    frame["text_raw"] = frame["request_http_request"].astype(str) + "\n" + frame["request_body"].astype(str)

    # URI+body가 같고 라벨만 다르면 keep='first'가 임의의 정답을 남기므로 전부 제외한다.
    label_counts = frame.groupby("text_raw")["label"].nunique()
    conflict_texts = label_counts.index[label_counts.gt(1)]
    conflicts = frame["text_raw"].isin(conflict_texts)
    stats["label_conflict_texts"] = len(conflict_texts)
    stats["label_conflict_rows"] = int(conflicts.sum())
    frame = frame.loc[~conflicts, OUTPUT_COLUMNS].copy()
    stats["n_returned"] = _counts(frame)
    return frame, stats


def load_srbh_4class_with_stats() -> tuple[pd.DataFrame, dict]:
    """감사 파일을 필수로 읽어 감사 없이 트랙이 생성되는 것을 막는다."""
    if not AUDIT_CSV.is_file():
        raise FileNotFoundError("감사 CSV가 없습니다. python src/data/audit_srbh_labels.py 를 실행하세요.")
    flags = pd.read_csv(AUDIT_CSV, dtype=str, keep_default_na=False, encoding="utf-8")
    if "row_id" not in flags or not flags["row_id"].str.fullmatch(r"[0-9]+").all():
        raise ValueError("감사 CSV에는 0 이상의 정수 row_id 컬럼이 필요합니다.")
    raw = pd.read_csv(RAW_DIR / FILENAME, dtype=str, keep_default_na=False, encoding="utf-8")
    return build_srbh_frame(raw, {int(value) for value in flags["row_id"]})


def load_srbh_4class() -> pd.DataFrame:
    """기존 TRACKS 로더 인터페이스에 맞춰 DataFrame만 반환한다."""
    return load_srbh_4class_with_stats()[0]
