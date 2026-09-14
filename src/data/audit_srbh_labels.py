"""고정된 v1 시그니처로 Normal 라벨을 감사하고 원본 행 번호를 보존한다."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

if __package__:
    from .download_srbh import FILENAME, PROJECT_ROOT, RAW_DIR
    from .preprocess import normalize_text
    from .profile_srbh import detect_label_columns
else:
    from download_srbh import FILENAME, PROJECT_ROOT, RAW_DIR
    from preprocess import normalize_text
    from profile_srbh import detect_label_columns

SCAN_FIELDS = (
    "request_http_request", "request_body", "request_cookie",
    "request_user_agent", "request_referer",
)

# 결과를 본 뒤 기준을 바꾸지 않도록 지시서의 패턴을 원형 그대로 고정한다.
SIGNATURES_V1 = {
    "union_select": {"class": "SQLi", "pattern": r"\bunion\b[\s\S]{0,20}?\bselect\b"},
    "tautology": {"class": "SQLi", "pattern": r'''['"\)]\s*\b(or|and)\b\s+['"(]?\s*\d+\s*['")]?\s*=\s*['"(]?\s*\d+'''},
    "time_based": {"class": "SQLi", "pattern": r"\b(sleep|benchmark|pg_sleep)\s*\(|\bwaitfor\s+delay\b"},
    "schema_probe": {"class": "SQLi", "pattern": r"\binformation_schema\b|\bsys(objects|columns)\b|@@version\b"},
    "error_based": {"class": "SQLi", "pattern": r"\b(extractvalue|updatexml)\s*\("},
    "stacked_query": {"class": "SQLi", "pattern": r";\s*(drop|insert|update|delete|exec|declare)\s"},
    "comment_terminator": {"class": "SQLi", "pattern": r'''['"]\s*(--|#|/\*)'''},
    "script_tag": {"class": "CodeInj", "pattern": r"<\s*script\b"},
    "event_handler": {"class": "CodeInj", "pattern": r"<[^>]*\bon(error|load|mouseover|focus|click|toggle)\s*="},
    "js_uri": {"class": "CodeInj", "pattern": r"javascript\s*:"},
    "dangerous_tag": {"class": "CodeInj", "pattern": r"<\s*(iframe|svg|object|embed)\b"},
    "js_sink": {"class": "CodeInj", "pattern": r"\b(alert|prompt|confirm)\s*\(|document\.(cookie|location)"},
    "php_code": {"class": "CodeInj", "pattern": r"<\?php|\b(eval|assert|base64_decode|system|passthru|shell_exec)\s*\("},
    "shell_chain": {"class": "CmdI", "pattern": r"(;|\|\|?|&&|\n)\s*(cat|ls|id|whoami|uname|wget|curl|nc|ping|bash|sh)\b"},
    "subshell": {"class": "CmdI", "pattern": r"\$\([^)]{1,100}\)|`[^`]{1,100}`"},
    "shell_path": {"class": "CmdI", "pattern": r"/bin/(ba)?sh\b|\bcmd\.exe\b|\bpowershell\b"},
}


def match_excerpt(text: str, pattern: re.Pattern) -> str:
    """첫 적중 구간 전체와 앞뒤 80자만 남겨 긴 요청의 불필요한 복제를 막는다."""
    match = pattern.search(text)
    if match is None:
        raise ValueError("발췌할 적중 구간이 없습니다.")
    return text[max(0, match.start() - 80):match.end() + 80]


def _join_hits(hits: pd.DataFrame) -> pd.Series:
    """정렬된 이름을 연결해 입력 컬럼 순서가 달라도 같은 표현을 만든다."""
    joined = pd.Series("", index=hits.index)
    for name in sorted(hits.columns):
        mask = hits[name]
        joined.loc[mask] = joined.loc[mask] + name + ";"
    return joined.str.rstrip(";")


def _scan(frame: pd.DataFrame, signatures: dict) -> tuple[pd.DataFrame, dict, dict]:
    missing = sorted(set(SCAN_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"필수 스캔 필드가 없습니다: {missing}")
    if not frame.index.is_unique or any(not isinstance(i, int) or i < 0 for i in frame.index):
        raise ValueError("원본 행 번호는 중복 없는 0 이상의 정수여야 합니다.")
    compiled = {name: re.compile(spec["pattern"], re.IGNORECASE) for name, spec in signatures.items()}
    pattern_hits = pd.DataFrame(False, index=frame.index, columns=list(signatures))
    field_hits = pd.DataFrame(False, index=frame.index, columns=list(SCAN_FIELDS))
    field_pattern_counts = {}
    candidates = {name: {} for name in signatures}
    for field in SCAN_FIELDS:
        # 필드별 디코딩은 한 번만 수행하고, 반복 문자열은 한 번 검색한 결과를 재사용한다.
        decoded = frame[field].map(normalize_text)
        unique_text = pd.Series(decoded.unique(), dtype=object)
        field_pattern_counts[field] = {}
        for name, pattern in compiled.items():
            # contains는 캡처 그룹을 반환하지 않는다. 고정 정규식을 수정하는 대신 이 안내만 숨긴다.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression.*", category=UserWarning)
                unique_hits = unique_text.str.contains(pattern, na=False)
            hits = decoded.isin(unique_text[unique_hits])
            pattern_hits[name] |= hits
            field_hits[field] |= hits
            field_pattern_counts[field][name] = int(hits.sum())
            # 각 필드의 앞 10개 합집합에는 요청 기준의 전역 앞 10개가 반드시 포함된다.
            for row_id in decoded.index[hits].sort_values()[:10]:
                entry = candidates[name].setdefault(int(row_id), {"row_id": int(row_id), "matches": []})
                entry["matches"].append({"field": field, "snippet": match_excerpt(decoded.at[row_id], pattern)})

    flagged = pattern_hits.any(axis=1)
    class_hits = pd.DataFrame({
        label: pattern_hits[[name for name, spec in signatures.items() if spec["class"] == label]].any(axis=1)
        for label in sorted({spec["class"] for spec in signatures.values()})
    }, index=frame.index)
    flags = pd.DataFrame({
        "row_id": frame.index[flagged],
        "audit_classes": _join_hits(class_hits.loc[flagged]).to_numpy(),
        "patterns": _join_hits(pattern_hits.loc[flagged]).to_numpy(),
        "fields": _join_hits(field_hits.loc[flagged]).to_numpy(),
    })
    n_flagged = int(flagged.sum())
    summary = {
        "n_scanned": len(frame), "n_flagged": n_flagged,
        "flag_rate": n_flagged / len(frame) if len(frame) else None,
        "class_combinations": {key.replace(";", "+"): int(value) for key, value in flags.audit_classes.value_counts().sort_index().items()},
        "pattern_counts": {name: int(pattern_hits[name].sum()) for name in signatures},
        "field_counts": {field: int(field_hits[field].sum()) for field in SCAN_FIELDS},
        "field_pattern_counts": field_pattern_counts,
    }
    samples = {name: [entries[i] for i in sorted(entries)[:10]] for name, entries in candidates.items()}
    return flags, summary, samples


def scan_frame(frame: pd.DataFrame, signatures: dict = SIGNATURES_V1) -> pd.DataFrame:
    """적중 요청만 반환하며 row_id와 반환 행 순서는 입력 인덱스를 따른다."""
    return _scan(frame, signatures)[0]


def audit_frame(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """감사와 공격 참고 집합을 분리해 참고 라벨이 감사 판정에 개입하지 않게 한다."""
    labels = detect_label_columns(frame)
    by_code = {name.split(" - ", 1)[0]: name for name in labels}
    if not {"000", "66", "242", "88", "248"} <= by_code.keys():
        raise ValueError("감사에 필요한 라벨 코드가 없습니다.")
    if not frame[labels].isin(["0", "1"]).all().all():
        raise ValueError("라벨에 문자열 0 또는 1 이외의 값이 있습니다.")
    active = frame[labels].eq("1")
    normal = active[by_code["000"]]
    targets = pd.DataFrame({
        "Normal": normal, "SQLi": active[by_code["66"]],
        "CodeInj": active[by_code["242"]],
        "CmdI": active[by_code["88"]] | active[by_code["248"]],
    })
    flags, summary, samples = _scan(frame.loc[normal], SIGNATURES_V1)
    assert normal.loc[flags.row_id].all(), "감사 목록에 Normal 이외의 원본 라벨이 포함됐습니다."
    reference = {}
    single_class = targets.sum(axis=1).eq(1)
    for label in ("SQLi", "CodeInj", "CmdI"):
        _, counts, _ = _scan(frame.loc[targets[label] & single_class], SIGNATURES_V1)
        reference[label] = {key: counts[key] for key in ("n_scanned", "n_flagged", "flag_rate", "pattern_counts")}
    return {
        "signature_version": "v1", "signatures": SIGNATURES_V1,
        "normal": summary, "attack_reference": reference, "samples": samples,
        "reference_wamm_mislabeled": 48522,
    }, flags.sort_values("row_id", ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="SR-BH 2020 Normal 라벨 시그니처 감사")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/results/srbh_label_audit.json", help="결과 JSON 경로")
    args = parser.parse_args()
    source = RAW_DIR / FILENAME
    if not source.is_file():
        print("원본 파일이 없습니다. python src/data/download_srbh.py 를 실행하세요.", file=sys.stderr)
        return 1
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8")
        started = time.perf_counter()
        result, flags = audit_frame(frame)
        result["scan_seconds"] = time.perf_counter() - started
        destination = PROJECT_ROOT / "data/processed/srbh_audit_flags.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        flags.to_csv(destination, index=False, encoding="utf-8")
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"normal": {key: result["normal"][key] for key in ("n_scanned", "n_flagged", "flag_rate", "class_combinations", "pattern_counts")},
                          "attack_reference": result["attack_reference"], "scan_seconds": result["scan_seconds"]}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"라벨 감사 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
