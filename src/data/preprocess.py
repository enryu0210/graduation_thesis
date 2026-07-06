"""
Phase 3-A — 전처리 및 데이터 분할 스크립트

목적:
    설계 문서 3장 "전처리 체크리스트"를 구현한다.
      1) URL 디코딩 · HTML 엔티티 디코딩으로 표기 정규화
         (단, RQ2 회피공격 난이도에 영향 → 'raw' 와 'decoded' 를 모두 컬럼으로 보존해
          "디코딩 적용 vs 미적용" ablation 을 나중에 할 수 있게 한다.)
      2) 완전 중복 페이로드 제거 (제거 전/후 수치 함께 보고)
      3) Train / Val / Test = 70 / 15 / 15, stratified split
         (중복 제거를 '분할 전'에 수행해 동일 페이로드가 train/test 에 나뉘는 누수(leakage) 방지)

데이터 트랙(라벨 공간이 다르므로 독립적으로 처리):
    - payload_4class : 공격 페이로드셋 (Normal/SQLi/XSS/CmdI) — RQ1 주 데이터셋
    - csic_binary    : CSIC 2010 HTTP 트래픽 (Normal/Anomalous) — 일반화/트래픽 트랙
                       (현 단계에서는 대표 텍스트로 URL 컬럼을 사용. 정확한 직렬화 방식은
                        docs/EDA_notes.md 의 메모대로 이후 실험 설계에서 확정)

원본 보존 원칙:
    data/raw 는 절대 수정하지 않는다. 산출물은 data/processed/ 에만 쓴다.

사용법:
    python src/data/preprocess.py                 # 두 트랙 모두 처리
    python src/data/preprocess.py payload_4class  # 특정 트랙만
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from urllib.parse import unquote_plus, urlsplit

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_CSV = (
    PROJECT_ROOT / "data" / "raw" / "payload_3class"
    / "SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv"
)
CSIC_CSV = PROJECT_ROOT / "data" / "raw" / "csic2010" / "csic_database.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NOTES_PATH = PROJECT_ROOT / "docs" / "03_preprocessing_notes.md"

PAYLOAD_LABEL_COLS = ["SQLInjection", "XSS", "CommandInjection", "Normal"]

# 분할 비율(train/val/test)과 난수 시드 — 재현성을 위해 고정.
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RANDOM_SEED = 42

# URL 디코딩 반복 상한: 이중/삼중 인코딩(%2520 등)을 풀되 무한루프를 막기 위해 상한을 둔다.
MAX_DECODE_ROUNDS = 3


# ---------------------------------------------------------------------------
# 텍스트 정규화 (디코딩)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """URL 디코딩과 HTML 엔티티 디코딩을 적용한 정규화 문자열을 만든다.

    - URL 디코딩은 이중 인코딩까지 고려해 '변화가 없을 때까지'(최대 MAX_DECODE_ROUNDS회) 반복한다.
    - 그 뒤 HTML 엔티티(&lt; 등)를 디코딩한다.
    - 원본은 건드리지 않고 새 문자열을 반환한다(호출부에서 raw 는 따로 보존).
    """
    decoded = text
    for _ in range(MAX_DECODE_ROUNDS):
        once = unquote_plus(decoded)
        if once == decoded:  # 더 이상 바뀌지 않으면 조기 종료
            break
        decoded = once
    return html.unescape(decoded)


# ---------------------------------------------------------------------------
# 원본 로딩 (트랙별로 (text_raw, label) 스키마로 통일)
# ---------------------------------------------------------------------------
def load_payload_4class() -> pd.DataFrame:
    """4-class 페이로드셋을 (text_raw, label) 스키마로 읽는다."""
    df = pd.read_csv(PAYLOAD_CSV, low_memory=False)
    label_numeric = df[PAYLOAD_LABEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return pd.DataFrame(
        {
            "text_raw": df["Sentence"].fillna("").astype(str),
            # 원-핫 → 단일 라벨 (Phase 1 검증에서 다중/무라벨 0 확인)
            "label": label_numeric.idxmax(axis=1),
        }
    )


def load_csic_binary() -> pd.DataFrame:
    """CSIC 2010 을 (text_raw, label) 스키마로 읽는다. label 은 Normal/Anomalous."""
    df = pd.read_csv(CSIC_CSV, low_memory=False)
    cls = pd.to_numeric(df["classification"], errors="coerce")
    return pd.DataFrame(
        {
            "text_raw": df["URL"].fillna("").astype(str),
            "label": cls.map({0: "Normal", 1: "Anomalous"}),
        }
    )


def load_payload_4class_csicnorm() -> pd.DataFrame:
    """RQ1 신뢰도 개선 트랙 — Normal 을 CSIC 실트래픽으로 교체한 4-class 셋.

    왜 만드나 (Phase 4 §7 편향 진단):
        기존 payload_4class 의 Normal 은 '영화 리뷰 등 영어 산문'이라, 모델이
        '공격이냐 정상이냐'가 아니라 '문장이냐 기호냐'를 배우는 shortcut 위험이 있었다.
        → 공격 3종(SQLi/XSS/CmdI)은 그대로 두고, Normal 만 CSIC 2010 정상 요청의
          '전체 쿼리스트링'으로 교체해 정상/공격이 같은 표현 공간(기호 범벅)에 있게 한다.

    왜 '단일 값'이 아니라 '전체 쿼리스트링'인가:
        단일 파라미터 값은 길이 중앙값이 9자에 불과해, 공격(75~600자)과 붙이면
        '짧으면 정상'이라는 새 shortcut 이 생긴다(측정 근거: docs/05_..._design 참조).
        전체 쿼리스트링(중앙값 71자)은 길이가 공격과 겹쳐 이 문제를 크게 완화한다.
    """
    # 1) 공격 3종: 기존 페이로드셋에서 산문 Normal 만 제외하고 그대로 사용
    payload = load_payload_4class()
    attacks = payload[payload["label"] != "Normal"].copy()

    # 2) Normal: CSIC 2010 정상(classification==0) 요청의 URL 에서 쿼리스트링만 추출
    csic = pd.read_csv(CSIC_CSV, low_memory=False)
    cls = pd.to_numeric(csic["classification"], errors="coerce")
    normal_urls = csic.loc[cls == 0, "URL"].fillna("").astype(str)
    query_strings = normal_urls.map(lambda u: urlsplit(u).query)
    # CSIC 의 URL 필드는 실제로 요청라인('... HTTP/1.1')이라 쿼리 끝에 프로토콜 꼬리표가
    # 붙는다. 이 ' HTTP/1.1' 은 공격 페이로드엔 전혀 없어 그대로 두면 모델이 이 한 토큰으로
    # Normal 을 분리하는 인공 shortcut 이 된다(진단 [3]에서 실측). → 반드시 제거한다.
    query_strings = query_strings.str.replace(r"\s+HTTP/\d(?:\.\d)?\s*$", "", regex=True)
    # 쿼리스트링이 없는 정적 GET(예: 이미지 요청)은 내용이 없어 제외
    query_strings = query_strings[query_strings.str.strip() != ""]
    normal_df = pd.DataFrame({"text_raw": query_strings.values, "label": "Normal"})

    # 완전 중복 제거·분할은 공통 clean_and_split 이 담당하므로 여기선 결합만 한다.
    return pd.concat([attacks, normal_df], ignore_index=True)


# ---------------------------------------------------------------------------
# 전처리 + 분할 (공통)
# ---------------------------------------------------------------------------
def clean_and_split(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """빈 값 제거 → 중복 제거 → 정규화 → stratified 70/15/15 분할을 수행한다.

    반환: (split 컬럼이 추가된 DataFrame, 리포트 라인 리스트)
    """
    report: list[str] = []
    n_start = len(df)
    report.append(f"- 시작 행 수: {n_start:,}")

    # 1) 빈/결측 텍스트 제거 (이미지로 만들 내용이 없으므로)
    df = df.copy()
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    empty_mask = df["text_raw"].str.strip() == ""
    df = df[~empty_mask]
    report.append(f"- 빈 문자열 제거: {int(empty_mask.sum()):,} 행 → 남은 {len(df):,}")

    # 라벨 결측 제거(예: CSIC classification 이 0/1 이 아닌 값)
    label_na = df["label"].isna()
    if int(label_na.sum()) > 0:
        df = df[~label_na]
        report.append(f"- 라벨 결측 제거: {int(label_na.sum()):,} 행 → 남은 {len(df):,}")

    # 2) 완전 중복 제거 (raw 기준). 분할 '전'에 수행해 train/test 누수를 방지.
    before = len(df)
    df = df.drop_duplicates(subset="text_raw", keep="first").reset_index(drop=True)
    removed = before - len(df)
    rate = removed / before if before else 0.0
    report.append(f"- 완전 중복 제거(raw 기준): {removed:,} 행 ({rate:.2%}) → 남은 {len(df):,}")

    # 3) 정규화 컬럼 추가 (raw 는 보존, decoded 를 새로 만듦 → ablation 대비)
    df["text_decoded"] = df["text_raw"].map(normalize_text)

    # 4) stratified 분할 (라벨 비율을 train/val/test 에서 동일하게 유지)
    #    2단계로 나눈다: 먼저 train vs (val+test), 그다음 (val+test)를 반으로.
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["label"], random_state=RANDOM_SEED,
    )
    # temp 안에서 val:test = 15:15 이므로 test 비중은 정확히 절반.
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["label"], random_state=RANDOM_SEED,
    )
    train_df, val_df, test_df = (d.assign(split=name) for d, name in
                                 [(train_df, "train"), (val_df, "val"), (test_df, "test")])
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 리포트: 분할별 크기 + 클래스 분포
    report.append("- 분할 결과(행 수 및 클래스 분포):")
    for name in ("train", "val", "test"):
        part = out[out["split"] == name]
        dist = ", ".join(f"{k}={v:,}" for k, v in part["label"].value_counts().sort_index().items())
        report.append(f"    - {name}: {len(part):,} ({len(part)/len(out):.1%}) | {dist}")

    return out, report


def save_track(name: str, out: pd.DataFrame) -> list[str]:
    """분할된 데이터를 data/processed/{name}_{split}.csv 로 저장한다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for split in ("train", "val", "test"):
        part = out[out["split"] == split][["text_raw", "text_decoded", "label"]]
        path = PROCESSED_DIR / f"{name}_{split}.csv"
        part.to_csv(path, index=False, encoding="utf-8")
        saved.append(f"    - `{path.relative_to(PROJECT_ROOT)}` ({len(part):,} 행)")
    return saved


# ---------------------------------------------------------------------------
# 트랙 정의 및 실행
# ---------------------------------------------------------------------------
TRACKS = {
    "payload_4class": load_payload_4class,
    "payload_4class_csicnorm": load_payload_4class_csicnorm,
    "csic_binary": load_csic_binary,
}

# 트랙별로 필요한 원본 파일(존재 확인용). csicnorm 은 두 원본을 모두 필요로 한다.
REQUIRED_FILES = {
    "payload_4class": [PAYLOAD_CSV],
    "payload_4class_csicnorm": [PAYLOAD_CSV, CSIC_CSV],
    "csic_binary": [CSIC_CSV],
}


def process_track(name: str) -> list[str]:
    """단일 트랙을 로딩→전처리→분할→저장하고 리포트 라인을 반환한다."""
    lines = [f"## 트랙: {name}", ""]
    df = TRACKS[name]()
    out, report = clean_and_split(df)
    lines += report
    lines.append("- 저장 파일:")
    lines += save_track(name, out)
    lines.append("")
    return lines


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    # 인자로 특정 트랙만 지정 가능. 없으면 전체.
    requested = sys.argv[1:] or list(TRACKS.keys())
    unknown = [t for t in requested if t not in TRACKS]
    if unknown:
        print(f"[중단] 알 수 없는 트랙: {unknown}. 가능한 값: {list(TRACKS.keys())}")
        return

    header = [
        "# 전처리·분할 결과 (preprocess.py)",
        "",
        "> `python src/data/preprocess.py` 실행 시 자동 생성됩니다.",
        "> 설계 3장 전처리 체크리스트(디코딩·중복제거·stratified split) 구현 결과입니다.",
        "",
        f"- 분할 비율: train {TRAIN_RATIO:.0%} / val {VAL_RATIO:.0%} / test {TEST_RATIO:.0%} "
        f"(stratified, seed={RANDOM_SEED})",
        "- 각 산출 CSV 컬럼: `text_raw`, `text_decoded`, `label` "
        "(raw/decoded 를 모두 보존해 디코딩 ablation 지원)",
        "",
        "---",
        "",
    ]

    body: list[str] = []
    for name in requested:
        # 입력 파일 존재 확인 (트랙마다 필요한 원본이 다름)
        missing = [p for p in REQUIRED_FILES[name] if not p.exists()]
        if missing:
            miss_str = ", ".join(f"`{p}`" for p in missing)
            body += [f"## 트랙: {name}", "", f"- [건너뜀] 입력 파일 없음: {miss_str}", ""]
            continue
        body += process_track(name)

    report = "\n".join(header + body) + "\n"
    NOTES_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[OK] 전처리 결과를 {NOTES_PATH.relative_to(PROJECT_ROOT)} 에 저장했습니다.")


if __name__ == "__main__":
    main()
