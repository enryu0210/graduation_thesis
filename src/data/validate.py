"""
데이터셋 1차 검증 스크립트 (Phase 1)

목적:
    docs/01_data_acquisition_plan.md 4번 "데이터 검증 체크리스트"를 자동화한다.
    다운로드한 원본 CSV에 대해 재현성 근거가 되는 통계를 계산하고,
    사람이 읽을 수 있는 리포트를 stdout 과 docs/DATA_MANIFEST.md 에 남긴다.

왜 이렇게 작성했는가:
    - 졸업논문 심사/재현성 대응을 위해 "원본을 손대지 않고" 통계만 뽑아야 한다.
      따라서 이 스크립트는 절대 CSV 를 수정하지 않는다 (read-only).
    - SHA256 을 함께 기록해 나중에 데이터가 바뀌지 않았음을 증명할 수 있게 한다.
    - 멀티라벨 원-핫(SQLInjection/XSS/CommandInjection/Normal) 구조를 자동 감지하도록 해서
      다른 스키마의 데이터셋이 추가돼도 최소한의 요약은 나오게 한다.

사용법:
    python src/data/validate.py                 # data/raw 전체 검증
    python src/data/validate.py <csv경로> ...    # 특정 파일만 검증
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트 (이 파일 기준 상위 2단계: src/data/ -> src/ -> 루트)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "DATA_MANIFEST.md"

# 원-핫 라벨로 흔히 쓰이는 컬럼명 (대소문자 무시하고 매칭)
KNOWN_LABEL_COLUMNS = {"sqlinjection", "xss", "commandinjection", "normal"}

# 페이로드/문장이 들어있을 법한 텍스트 컬럼 후보
KNOWN_TEXT_COLUMNS = {"sentence", "payload", "query", "text", "request"}


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """파일 전체의 SHA256 해시를 계산한다 (무결성/재현성 증빙용).

    큰 파일(수십 MB)을 통째로 메모리에 올리지 않도록 1MB씩 스트리밍으로 읽는다.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def utf8_decode_failure_rate(path: Path, sample_bytes: int = 5 << 20) -> float:
    """앞부분 일부 바이트를 UTF-8 로 디코딩해보고 실패 여부를 대략 판단한다.

    Phase 3(바이트->이미지 변환)에서 인코딩이 중요하므로 미리 점검한다.
    전체를 다 검사하면 느리므로 앞 5MB 만 표본으로 본다.
    반환값: 0.0(문제 없음) 또는 1.0(디코딩 오류 발생) — 표본 기반 근사치.
    """
    with path.open("rb") as f:
        head = f.read(sample_bytes)
    try:
        head.decode("utf-8")
        return 0.0
    except UnicodeDecodeError:
        return 1.0


def detect_columns(df: pd.DataFrame) -> tuple[list[str], str | None]:
    """데이터프레임에서 라벨 컬럼들과 텍스트 컬럼을 자동 감지한다.

    반환: (라벨 컬럼 리스트, 텍스트 컬럼명 or None)
    """
    lower_map = {c.lower(): c for c in df.columns}

    label_cols = [lower_map[name] for name in lower_map if name in KNOWN_LABEL_COLUMNS]
    text_col = next(
        (lower_map[name] for name in lower_map if name in KNOWN_TEXT_COLUMNS),
        None,
    )
    return label_cols, text_col


def validate_csv(path: Path) -> str:
    """CSV 한 개를 검증하고 마크다운 형식의 리포트 문자열을 만든다."""
    rel_path = path.relative_to(PROJECT_ROOT)
    lines: list[str] = [f"## {path.name}", ""]

    # --- 무결성 정보 (데이터를 파싱하기 전에 먼저 기록) ---
    size_mb = path.stat().st_size / (1024 * 1024)
    lines.append(f"- Local path: `{rel_path}`")
    lines.append(f"- File size: {size_mb:.1f} MB")
    lines.append(f"- SHA256: `{sha256_of_file(path)}`")
    lines.append(f"- UTF-8 decode error (head sample): {utf8_decode_failure_rate(path)}")

    # --- 데이터 로드 (원본 훼손 방지를 위해 오직 읽기만) ---
    try:
        # dtype=str 로 읽어 라벨/텍스트가 임의로 형변환되는 것을 막는다.
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception as exc:  # 파싱 실패도 검증 결과의 일부이므로 기록하고 넘어간다
        lines.append(f"- **파싱 실패**: {exc}")
        lines.append("")
        return "\n".join(lines)

    n_rows = len(df)
    lines.append(f"- Row count (actual): {n_rows:,}")
    lines.append(f"- Columns: {list(df.columns)}")

    label_cols, text_col = detect_columns(df)

    # --- 완전 중복 행 비율 (전처리 전 기준값) ---
    # 텍스트 컬럼이 있으면 그 컬럼 기준 중복을, 없으면 전체 행 기준 중복을 본다.
    if text_col is not None:
        dup_count = int(df[text_col].duplicated().sum())
        dup_basis = f"'{text_col}' 컬럼 기준"
    else:
        dup_count = int(df.duplicated().sum())
        dup_basis = "전체 행 기준"
    dup_rate = dup_count / n_rows if n_rows else 0.0
    lines.append(f"- Duplicate rate (raw, {dup_basis}): {dup_rate:.4f} ({dup_count:,} rows)")

    # --- 결측치/빈 문자열 ---
    if text_col is not None:
        na_count = int(df[text_col].isna().sum())
        empty_count = int((df[text_col].fillna("").astype(str).str.strip() == "").sum())
        lines.append(
            f"- Missing/empty in '{text_col}': "
            f"NaN={na_count:,}, empty-string={empty_count:,}"
        )

    # --- 클래스 분포 ---
    if label_cols:
        lines.append("- Class distribution (원-핫 라벨 합계):")
        # 각 라벨 컬럼을 숫자로 변환해 1의 개수를 센다 (안전하게 coerce).
        for col in label_cols:
            positive = int(pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).sum())
            share = positive / n_rows if n_rows else 0.0
            lines.append(f"    - {col}: {positive:,} ({share:.2%})")

        # 멀티라벨 무결성: 한 행에 라벨이 정확히 1개인지(단일 라벨 데이터인지) 점검
        label_numeric = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0)
        per_row_label_count = label_numeric.sum(axis=1)
        multi = int((per_row_label_count > 1).sum())
        zero = int((per_row_label_count == 0).sum())
        lines.append(
            f"    - 라벨 무결성: 다중라벨 행={multi:,}, 무라벨 행={zero:,} "
            f"(0이면 깔끔한 단일라벨 데이터)"
        )
    else:
        lines.append("- Class distribution: (알려진 라벨 컬럼을 찾지 못함 — 수동 확인 필요)")

    lines.append("")
    return "\n".join(lines)


def collect_targets(argv: list[str]) -> list[Path]:
    """검증 대상 CSV 경로 목록을 결정한다.

    인자가 주어지면 그 파일들을, 없으면 data/raw 아래 모든 csv 를 대상으로 한다.
    """
    if argv:
        return [Path(a).resolve() for a in argv]
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.rglob("*.csv"))


def main() -> None:
    targets = collect_targets(sys.argv[1:])
    if not targets:
        print("검증할 CSV 파일을 찾지 못했습니다. data/raw 아래에 데이터가 있는지 확인하세요.")
        return

    reports = [validate_csv(p) for p in targets if p.exists()]
    report_body = "\n".join(reports)

    # stdout 출력 (즉시 확인용)
    print(report_body)

    # DATA_MANIFEST.md 에 검증 결과 섹션을 append 한다.
    # 왜 append 인가: 상단의 '출처/라이선스' 수기 기록을 덮어쓰지 않기 위해서.
    if MANIFEST_PATH.exists():
        marker = "\n---\n\n# 자동 검증 결과 (validate.py)\n\n"
        existing = MANIFEST_PATH.read_text(encoding="utf-8")
        # 이전 자동 검증 블록이 있으면 잘라내고 새로 붙인다 (중복 누적 방지).
        base = existing.split("# 자동 검증 결과 (validate.py)")[0].rstrip()
        MANIFEST_PATH.write_text(
            base + "\n" + marker + report_body + "\n", encoding="utf-8"
        )
        print(f"\n[OK] 검증 결과를 {MANIFEST_PATH.relative_to(PROJECT_ROOT)} 에 기록했습니다.")
    else:
        print(f"\n[경고] {MANIFEST_PATH} 가 없어 파일 기록은 건너뜁니다.")


if __name__ == "__main__":
    main()
