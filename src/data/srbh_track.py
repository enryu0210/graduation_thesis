"""SR-BH의 감사 제외와 라벨 충돌 제거를 공통 분할보다 먼저 적용한다."""

from __future__ import annotations

import re
from urllib.parse import unquote_plus

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

# --------------------------------------------------------------------------
# E17 라벨 보정 — URL 경로의 명령 삽입(docs/14 §8.45, 규칙 고정 docs/paper_revision/03_prereg_v2.md §E17)
# SR-BH 는 URL 경로에 넣은 셸 명령 삽입(cat /etc/passwd, sleep 15, 셸쇼크)에 CmdI 라벨을 한 번도 붙이지 않았다
# — SQLi(17,643행) 또는 Normal(8,711행, 거의 스캐너 층)로 붙었고, 같은 변형도 두 라벨로 갈린다. 쿼리·바디의 같은
# 명령은 CmdI 다. 그대로 두면 모델이 "주입 위치·변형"을 외워야 맞히므로, 의미 기준 라벨 판을 만든다.
# 정규식은 휴리스틱이다 — SQL 흔적이 조금이라도 있으면 보정하지 않는 쪽(보수적)으로 짰다.
# --------------------------------------------------------------------------
# 셸 명령 흔적. 'id'·'ls' 같은 짧은 단어는 쿼리의 '&id=' 와 겹쳐 뺐다. SQL 의 sleep(5) 와 갈리게 sleep 뒤 괄호는 받지 않는다.
CMD_PATTERN = re.compile(
    r"(cat\s+/etc/passwd|\bsleep(?:\s+|\$\{?\w*\}?\s*)\d+"
    r"|[;&|`]\s*(?:cat|whoami|ping|echo|uname|nslookup)\b|\(\)\s*\{\s*:;\s*\}|\$\(|`[^`]+`)", re.I)
SQL_PATTERN = re.compile(
    r"(\bunion\b|\bselect\b|\b(?:or|and)\b\s+[\w'\"]+\s*(?:=|<|>|like)|--|/\*|\bsleep\s*\(|benchmark\s*\("
    r"|waitfor\s+delay|information_schema|extractvalue|updatexml|\border\s+by\b|pg_sleep|\bxor\b)", re.I)


def _decode(text: str) -> str:
    # 스캐너 페이로드는 이중 인코딩(%2527 등)도 있어 두 번 푼다.
    return unquote_plus(unquote_plus(text))


# 보정 대상 원 라벨. CodeInjection 은 경로 셸 명령 0행이라 넣지 않는다(넣으면 규칙만 넓어지고 바뀌는 행은 없다).
RELABEL_FROM = ("SQLInjection", "Normal")


def path_cmdi_mask(texts: list[str], labels: list[str]) -> list[bool]:
    """SQLi·Normal 라벨이면서 경로에 셸 명령이 있고 F2 전체에 SQL 흔적이 없는 행 → True(CmdI 로 보정할 행).

    texts 는 F2 원문(text_raw = URI + '\\n' + body). 경로 = 첫 줄의 '?' 앞.
    """
    if len(texts) != len(labels):
        raise ValueError("texts 와 labels 의 길이가 다릅니다.")
    mask = []
    for text, label in zip(texts, labels):
        if label not in RELABEL_FROM:
            mask.append(False)
            continue
        path = _decode(text.split("\n", 1)[0].partition("?")[0])
        mask.append(bool(CMD_PATTERN.search(path)) and not SQL_PATTERN.search(_decode(text)))
    return mask


FIELD_COMBINATIONS = {
    "F1": ("request_http_request",),
    "F2": ("request_http_request", "request_body"),
    "F3": ("request_http_request", "request_body", "request_cookie"),
    "F3M": ("request_http_request", "request_body", "request_cookie"),
    "F4": ("request_http_request", "request_body", "request_cookie", "request_user_agent"),
    "UC": ("request_user_agent", "request_cookie"),
}
MASKED_COLUMNS = {"F3M": ("request_cookie",)}


def compose_fields(frame: pd.DataFrame, combination: str) -> list[str]:
    """빈 필드의 구분자까지 보존해 조합 간 바이트 표현을 고정한다."""
    if combination not in FIELD_COMBINATIONS:
        raise ValueError(f"알 수 없는 필드 조합: {combination}")
    columns = FIELD_COMBINATIONS[combination]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"필드 조합에 필요한 컬럼이 없습니다: {missing}")
    values = frame[list(columns)].fillna("").astype(str)
    # 쿠키의 존재·문자 길이는 보존해 내용 자체의 기여만 분리한다.
    for column in MASKED_COLUMNS.get(combination, ()):
        values[column] = values[column].map(lambda value: "x" * len(value))
    return ["\n".join(row) for row in values.itertuples(index=False, name=None)]


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
