"""SR-BH 원본의 라벨 구조만 측정하며 매핑 선택이나 원문 변환은 하지 않는다."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

if __package__:
    from .download_srbh import EXPECTED_MD5, EXPECTED_SIZE, FILENAME, PROJECT_ROOT, RAW_DIR
else:
    from download_srbh import EXPECTED_MD5, EXPECTED_SIZE, FILENAME, PROJECT_ROOT, RAW_DIR

TEXT_FIELDS = (
    "request_http_request", "request_body", "request_cookie", "request_user_agent"
)


def detect_label_columns(frame: pd.DataFrame) -> list[str]:
    """배포본의 이름과 순서를 보존하되 예상과 다른 스키마는 거부한다."""
    labels = [name for name in frame.columns if isinstance(name, str) and re.match(r"^\d+ - ", name)]
    if len(labels) != 14:
        raise ValueError(f"라벨 컬럼은 정확히 14개여야 합니다: 실제 {len(labels)}개")
    codes = [name.split(" - ", 1)[0] for name in labels]
    if len(set(codes)) != len(codes):
        raise ValueError("라벨 코드가 중복되어 있습니다.")
    return labels


def mapping_counts(targets: pd.DataFrame) -> dict[str, int]:
    """라벨 수가 아닌 대상 클래스 수로 중복을 판단하여 88+248을 한 클래스로 센다."""
    cardinality = targets.sum(axis=1)
    counts = {name: int((targets[name] & cardinality.eq(1)).sum()) for name in targets}
    counts.update(ambiguous=int(cardinality.ge(2).sum()), excluded=int(cardinality.eq(0).sum()))
    return counts


def profile_labels(frame: pd.DataFrame) -> dict:
    """파일 접근 없이 문자열 DataFrame에서 관측된 1과 비정상 값을 별도로 집계한다."""
    labels = detect_label_columns(frame)
    by_code = {name.split(" - ", 1)[0]: name for name in labels}
    missing = sorted(set(("000", "66", "242", "88", "248")) - by_code.keys())
    if missing:
        raise ValueError(f"필수 라벨 코드가 없습니다: {missing}")
    if any(name not in frame.columns for name in TEXT_FIELDS):
        raise ValueError("빈 문자열 비율 측정에 필요한 텍스트 컬럼이 없습니다.")

    values = frame[labels]
    invalid = ~values.isin(["0", "1"])
    # 잘못된 값을 0으로 보정하지 않고 건수와 원래 값을 결과에 명시한다.
    invalid_values = {
        name: {str(value): int(count) for value, count in values.loc[invalid[name], name].value_counts(dropna=False).items()}
        for name in labels
    }
    active = values.eq("1")
    label_counts = {name: int(active[name].sum()) for name in labels}
    cardinality = active.sum(axis=1).value_counts().sort_index()
    # bool 행렬 곱은 건수가 아니며 작은 정수형은 넘칠 수 있어 int64를 사용한다.
    matrix = active.to_numpy(dtype="int64")
    cooccurrence = matrix.T @ matrix
    normal = active[by_code["000"]]
    cmd88, cmd248 = active[by_code["88"]], active[by_code["248"]]
    only248 = cmd248 & ~cmd88
    both = int((cmd88 & cmd248).sum())
    targets = pd.DataFrame({
        "Normal": normal, "SQLi": active[by_code["66"]],
        "CodeInj": active[by_code["242"]], "CmdI": cmd88 | cmd248,
    })
    scenario_a = mapping_counts(targets)
    targets["CmdI"] = cmd88
    return {
        "n_rows": len(frame), "n_columns": len(frame.columns), "label_columns": labels,
        "label_counts": label_counts,
        "cardinality": {str(k): int(v) for k, v in cardinality.items()},
        "cooccurrence": {name: {other: int(cooccurrence[i, j]) for j, other in enumerate(labels)} for i, name in enumerate(labels)},
        "normal_conflict": int((normal & active.drop(columns=by_code["000"]).any(axis=1)).sum()),
        "cmd_injection_structure": {
            "only_88": int((cmd88 & ~cmd248).sum()), "only_248": int(only248.sum()),
            "both_88_248": both,
            # 조건 사건이 없을 때 0이라는 확률을 만들어내지 않는다.
            "p_248_given_88": both / int(cmd88.sum()) if cmd88.any() else None,
            "p_88_given_248": both / int(cmd248.sum()) if cmd248.any() else None,
            "only_248_other_label_counts": {name: int(active.loc[only248, name].sum()) for name in labels if name not in (by_code["88"], by_code["248"])},
        },
        "mapping_scenarios": {"A_merge_248": scenario_a, "B_drop_248_only": mapping_counts(targets)},
        "text_field_empty_rate": {name: float(frame[name].eq("").mean()) if len(frame) else None for name in TEXT_FIELDS},
        "invalid_label_values": {"n_cells": int(invalid.to_numpy().sum()), "n_rows": int(invalid.any(axis=1).sum()), "by_label": invalid_values},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SR-BH 2020 라벨 구조 실측")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/results/srbh_label_profile.json", help="결과 JSON 경로")
    args = parser.parse_args()
    source = RAW_DIR / FILENAME
    if not source.is_file():
        print("원본 파일이 없습니다. python src/data/download_srbh.py 를 실행하세요.", file=sys.stderr)
        return 1
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8")
        result = profile_labels(frame)
        result["source"] = {"filename": FILENAME, "size_bytes": source.stat().st_size, "expected_size_bytes": EXPECTED_SIZE, "EXPECTED_MD5": EXPECTED_MD5}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary = {key: result[key] for key in ("n_rows", "n_columns", "label_columns", "label_counts", "cmd_injection_structure", "mapping_scenarios")}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"라벨 측정 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
