"""
Phase 2 — 데이터 탐색(EDA) 및 이미지 변환 폭(W) 결정 스크립트

목적:
    설계 문서(web_attack_image_cnn_thesis_design.md) 4장 Step 2 는
    페이로드를 2D 이미지로 리셰이프할 때 "고정 폭 W" 를 데이터셋의
    페이로드 길이 분포(EDA)를 근거로 정하라고 요구한다.
    이 스크립트는 그 근거를 자동으로 산출한다.

무엇을 계산하는가:
    1. 각 페이로드(문자열)를 UTF-8 바이트로 인코딩했을 때의 "바이트 길이" 분포
       - 전체 / 클래스별(SQLi, XSS, CmdI, Normal) 요약 통계와 백분위수
    2. 후보 정사각 이미지 크기(예: 32x32=1024B)별로
       "잘림(truncation) 없이 담기는 샘플 비율(coverage)"
    3. 위 결과를 바탕으로 한 권장 W(=정사각 한 변) 자동 추천
    4. 분포 히스토그램 PNG 저장 (논문 4장 그림 자료)
    5. 사람이 읽는 리포트를 docs/EDA_notes.md 로 저장

왜 바이트 길이인가:
    이미지 변환은 "문자 수"가 아니라 "UTF-8 바이트 수" 기준으로 이뤄지므로
    (Nataraj 방식: 1 byte -> 1 pixel), 이미지 용량 산정도 바이트 기준이어야 한다.

원본 보존 원칙:
    이 스크립트는 data/raw 의 CSV 를 절대 수정하지 않는다 (read-only).

사용법:
    python src/data/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# matplotlib 은 GUI 없는 환경(서버/CI)에서도 그림을 파일로만 저장하도록 Agg 백엔드 사용.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 경로 설정 (이 파일 기준 상위 2단계가 프로젝트 루트) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_CSV = (
    PROJECT_ROOT
    / "data" / "raw" / "payload_3class"
    / "SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv"
)
CSIC_CSV = PROJECT_ROOT / "data" / "raw" / "csic2010" / "csic_database.csv"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "eda"
NOTES_PATH = PROJECT_ROOT / "docs" / "EDA_notes.md"

# 4-class 페이로드셋의 원-핫 라벨 컬럼 (순서 = 리포트 출력 순서)
PAYLOAD_LABEL_COLS = ["SQLInjection", "XSS", "CommandInjection", "Normal"]

# 후보 정사각 이미지 크기(한 변). 용량(byte) = side*side.
# 악성코드 이미지화 관례(32x32 등)와 커버리지 절충을 보기 위한 후보들.
CANDIDATE_SIDES = [16, 24, 32, 48, 64]

# 요약에 쓸 백분위수
PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# 권장 W 선정 기준: "잘림 없이 담기는 비율"이 이 값 이상인 가장 작은 정사각 크기를 고른다.
# 너무 크게 잡으면 대부분 zero-padding 이 되어 학습 신호가 희석되므로 절충값(95%)을 기본으로 둔다.
COVERAGE_TARGET = 0.95


def byte_lengths(text_series: pd.Series) -> pd.Series:
    """문자열 시리즈를 UTF-8 바이트 길이 시리즈로 변환한다.

    - 결측치(NaN)는 빈 문자열로 처리해 길이 0 으로 센다.
    - errors='replace': 인코딩 불가 문자가 있어도 예외로 죽지 않고 대체 문자로 처리
      (EDA 단계에서는 "죽지 않고 전체 분포를 보는 것"이 더 중요하기 때문).
    """
    cleaned = text_series.fillna("").astype(str)
    return cleaned.str.encode("utf-8", errors="replace").str.len()


def length_stats(lengths: pd.Series) -> dict[str, float]:
    """바이트 길이 분포의 요약 통계를 딕셔너리로 반환한다."""
    stats: dict[str, float] = {
        "count": int(len(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
    }
    for q in PERCENTILES:
        stats[f"p{q}"] = float(np.percentile(lengths, q))
    return stats


def coverage_by_size(lengths: pd.Series, sides: list[int]) -> list[tuple[int, int, float]]:
    """후보 정사각 크기별로 '잘림 없이 담기는 샘플 비율'을 계산한다.

    반환: [(side, capacity_bytes, coverage_ratio), ...]
    """
    n = len(lengths)
    table: list[tuple[int, int, float]] = []
    for side in sides:
        capacity = side * side
        coverage = float((lengths <= capacity).mean()) if n else 0.0
        table.append((side, capacity, coverage))
    return table


def recommend_side(lengths: pd.Series, sides: list[int], target: float) -> int:
    """coverage 가 target 이상인 가장 작은 정사각 크기를 권장 W 로 고른다.

    어떤 후보도 target 을 못 넘으면 가장 큰 후보를 반환한다(최선의 커버리지).
    """
    for side in sorted(sides):
        capacity = side * side
        if (lengths <= capacity).mean() >= target:
            return side
    return max(sides)


def derive_payload_labels(df: pd.DataFrame) -> pd.Series:
    """원-핫 라벨(4컬럼)에서 각 행의 단일 클래스명을 뽑아낸다.

    Phase 1 검증에서 다중/무라벨 행이 0 임을 확인했으므로 idxmax 로 안전하게 단일 라벨화한다.
    """
    label_numeric = df[PAYLOAD_LABEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return label_numeric.idxmax(axis=1)


def save_histogram(
    lengths: pd.Series,
    title: str,
    out_path: Path,
    clip_percentile: float = 99.0,
) -> None:
    """바이트 길이 히스토그램을 저장한다.

    롱테일(극단적으로 긴 소수 페이로드) 때문에 그림이 뭉개지는 것을 막기 위해
    상위 clip_percentile 지점까지만 x축에 표시한다(분포 형태 가독성 우선).
    라벨은 폰트 호환성을 위해 영어로 둔다(한글 폰트 미설치 환경 대비).
    """
    upper = float(np.percentile(lengths, clip_percentile))
    plt.figure(figsize=(8, 4.5))
    plt.hist(lengths.clip(upper=upper), bins=60, color="#4C72B0", edgecolor="white")
    plt.axvline(upper, color="#C44E52", linestyle="--", linewidth=1,
                label=f"p{clip_percentile:g} = {upper:.0f} B")
    plt.title(title)
    plt.xlabel(f"UTF-8 byte length (clipped at p{clip_percentile:g})")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_class_boxplot(
    lengths: pd.Series, labels: pd.Series, title: str, out_path: Path,
    clip_percentile: float = 99.0,
) -> None:
    """클래스별 바이트 길이 분포를 박스플롯으로 비교 저장한다."""
    upper = float(np.percentile(lengths, clip_percentile))
    order = PAYLOAD_LABEL_COLS
    data = [lengths[labels == c].clip(upper=upper).values for c in order]

    plt.figure(figsize=(8, 4.5))
    # tick_labels/labels 인자는 matplotlib 버전마다 이름이 달라 경고가 나므로,
    # 버전에 무관하게 동작하도록 xticks 로 라벨을 직접 지정한다.
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(order) + 1), order)
    plt.title(title)
    plt.ylabel(f"UTF-8 byte length (clipped at p{clip_percentile:g})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def format_stats_block(name: str, stats: dict[str, float]) -> list[str]:
    """요약 통계 딕셔너리를 마크다운 리스트 라인으로 변환한다."""
    lines = [f"- **{name}** (n={stats['count']:,})"]
    lines.append(
        f"    - min={stats['min']}, mean={stats['mean']:.1f}, "
        f"std={stats['std']:.1f}, max={stats['max']:,}"
    )
    pct = ", ".join(f"p{q:g}={stats[f'p{q}']:.0f}" for q in PERCENTILES)
    lines.append(f"    - 백분위수: {pct}")
    return lines


def format_coverage_table(table: list[tuple[int, int, float]]) -> list[str]:
    """coverage 표를 마크다운 표로 변환한다."""
    lines = ["| 이미지 크기 | 용량(byte) | 무손실 커버리지 |", "|---|---|---|"]
    for side, cap, cov in table:
        lines.append(f"| {side}×{side} | {cap:,} | {cov:.2%} |")
    return lines


def analyze_payload() -> list[str]:
    """4-class 페이로드셋을 분석하고 리포트 섹션(마크다운 라인들)을 반환한다."""
    df = pd.read_csv(PAYLOAD_CSV, low_memory=False)
    lengths = byte_lengths(df["Sentence"])
    labels = derive_payload_labels(df)

    lines: list[str] = ["## 1. 공격 페이로드셋 (4-class) 바이트 길이 분포", ""]
    lines.append(f"- 대상 파일: `{PAYLOAD_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    # 전체 + 클래스별 요약 통계
    lines += format_stats_block("전체", length_stats(lengths))
    for c in PAYLOAD_LABEL_COLS:
        lines += format_stats_block(c, length_stats(lengths[labels == c]))
    lines.append("")

    # 커버리지 표
    lines.append("### 후보 이미지 크기별 무손실 커버리지 (전체)")
    lines.append("")
    lines += format_coverage_table(coverage_by_size(lengths, CANDIDATE_SIDES))
    lines.append("")

    # 권장 W 선정 근거가 되는, 권장 크기에서의 "클래스별" 커버리지
    rec_side = recommend_side(lengths, CANDIDATE_SIDES, COVERAGE_TARGET)
    cap = rec_side * rec_side
    lines.append(
        f"### 권장 크기 {rec_side}×{rec_side}({cap:,}B) 에서의 클래스별 무손실 커버리지"
    )
    lines.append("")
    lines.append("| 클래스 | 무손실 커버리지 | 잘리는 비율 |")
    lines.append("|---|---|---|")
    for c in PAYLOAD_LABEL_COLS:
        b = lengths[labels == c]
        cov = float((b <= cap).mean())
        lines.append(f"| {c} | {cov:.2%} | {1 - cov:.2%} |")
    lines.append("")

    # 그림 저장
    save_histogram(
        lengths, "Payload UTF-8 byte length distribution (4-class)",
        FIG_DIR / "payload_bytelen_hist.png",
    )
    save_class_boxplot(
        lengths, labels, "Payload byte length by class",
        FIG_DIR / "payload_bytelen_by_class.png",
    )
    lines.append("![payload 길이 분포](figures/eda/payload_bytelen_hist.png)")
    lines.append("")
    lines.append("![클래스별 길이 분포](figures/eda/payload_bytelen_by_class.png)")
    lines.append("")

    return lines


def analyze_csic() -> list[str]:
    """CSIC 2010 의 URL 바이트 길이 분포를 분석해 리포트 섹션을 반환한다."""
    df = pd.read_csv(CSIC_CSV, low_memory=False)
    lengths = byte_lengths(df["URL"])
    # classification: 0=Normal, 1=Anomalous (Phase 1 검증에서 확인)
    cls = pd.to_numeric(df["classification"], errors="coerce").fillna(-1).astype(int)

    lines: list[str] = ["## 2. CSIC 2010 HTTP 요청(URL) 바이트 길이 분포", ""]
    lines.append(f"- 대상 파일: `{CSIC_CSV.relative_to(PROJECT_ROOT)}` (컬럼: `URL`)")
    lines.append(
        "- 참고: CSIC 을 이미지화할 때 정확히 무엇을(URL만? URL+content? 전체 요청 직렬화?)"
        " 담을지는 Phase 3 에서 확정한다. 여기서는 우선 URL 길이만 본다."
    )
    lines.append("")

    lines += format_stats_block("URL 전체", length_stats(lengths))
    lines += format_stats_block("Normal(0)", length_stats(lengths[cls == 0]))
    lines += format_stats_block("Anomalous(1)", length_stats(lengths[cls == 1]))
    lines.append("")

    lines.append("### 후보 이미지 크기별 무손실 커버리지 (URL)")
    lines.append("")
    lines += format_coverage_table(coverage_by_size(lengths, CANDIDATE_SIDES))
    lines.append("")

    save_histogram(
        lengths, "CSIC 2010 URL UTF-8 byte length distribution",
        FIG_DIR / "csic_url_bytelen_hist.png",
    )
    lines.append("![CSIC URL 길이 분포](figures/eda/csic_url_bytelen_hist.png)")
    lines.append("")

    return lines


def build_recommendation() -> list[str]:
    """두 데이터셋을 종합해 권장 W 결론 섹션을 만든다.

    이미지 변환 병목은 (URL 이 짧은) CSIC 이 아니라 페이로드셋이므로,
    W 는 페이로드셋 분포를 기준으로 정한다.
    """
    df = pd.read_csv(PAYLOAD_CSV, low_memory=False)
    lengths = byte_lengths(df["Sentence"])
    rec_side = recommend_side(lengths, CANDIDATE_SIDES, COVERAGE_TARGET)
    cap = rec_side * rec_side
    cov = float((lengths <= cap).mean())

    lines: list[str] = ["## 3. 결론 — 권장 이미지 변환 폭(W)", ""]
    lines.append(
        f"- **권장 W = {rec_side} (목표 이미지 {rec_side}×{rec_side}, 용량 {cap:,} byte)**"
    )
    lines.append(
        f"    - 근거: 페이로드셋 전체의 {cov:.2%} 가 잘림 없이 담김 "
        f"(커버리지 목표 {COVERAGE_TARGET:.0%} 를 만족하는 가장 작은 정사각 크기)."
    )
    lines.append(
        "    - 변환 규칙(설계 4장): 높이 = ceil(바이트수 / W), "
        f"{rec_side}×{rec_side} 보다 짧으면 zero-padding, 길면 truncation."
    )
    lines.append("")
    # 권장 크기 기준으로 "더 작은 관례 크기"와 "더 큰 커버리지 크기"를 ablation 후보로 제시.
    smaller = max([s for s in CANDIDATE_SIDES if s < rec_side], default=rec_side)
    larger = min([s for s in CANDIDATE_SIDES if s > rec_side], default=rec_side)

    lines.append("### 유의사항 / 리스크")
    lines.append(
        "- **CommandInjection 롱테일**: CmdI 는 다른 클래스보다 페이로드가 훨씬 길어"
        " 권장 크기에서도 잘림이 이 클래스에 집중된다(위 '클래스별 커버리지' 표 참고)."
        " 잘림이 CmdI recall 을 떨어뜨릴 수 있으므로 이 표를 논문에 함께 보고하고,"
        f" ablation 으로 {smaller}×{smaller}(더 작고 관례적)와 {larger}×{larger}(더 높은 커버리지)를 비교할 것."
    )
    lines.append(
        "- **CSIC 은 URL 만 보면 매우 짧다**(대부분 32×32 로 충분). 페이로드셋과 이미지 크기를"
        " 통일할지, 데이터셋별로 다르게 할지는 Phase 3 에서 실험 설계와 함께 결정한다."
    )
    lines.append(
        "- 여기 수치는 **중복 제거 전(raw)** 기준이다. 전처리(중복 제거·디코딩) 후 분포가"
        " 달라질 수 있으므로, 전처리 확정 후 이 스크립트를 재실행해 W 를 재검토할 것."
    )
    lines.append("")
    return lines


def main() -> None:
    # Windows 기본 콘솔(cp949)에서 한글/em대시 출력 시 UnicodeEncodeError 가 나므로
    # stdout 을 UTF-8 로 재설정한다(파일 저장은 이미 UTF-8 이라 영향 없음).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # 재설정이 불가능한 환경이면 무시(파일 저장은 정상 동작)

    # 입력 파일 존재 확인 (없으면 명확히 안내하고 종료 — 조용히 실패하지 않음)
    missing = [p for p in (PAYLOAD_CSV, CSIC_CSV) if not p.exists()]
    if missing:
        print("[중단] 다음 입력 파일을 찾지 못했습니다:")
        for p in missing:
            print(f"  - {p}")
        print("Phase 1 데이터 확보가 끝났는지 확인하세요.")
        return

    header = [
        "# EDA Notes — 페이로드 길이 분포 및 이미지 변환 폭(W) 결정",
        "",
        "> 이 문서는 `python src/data/eda.py` 실행 시 자동 생성됩니다 (수기 편집 시 재실행하면 덮어써짐).",
        "> 설계 문서 4장 Step 2(고정 폭 W 결정)의 데이터 근거 자료입니다.",
        "",
        f"- 커버리지 목표: {COVERAGE_TARGET:.0%} (잘림 없이 담기는 샘플 비율 기준)",
        f"- 후보 이미지 크기(한 변): {CANDIDATE_SIDES}",
        "",
        "---",
        "",
    ]

    sections = (
        header
        + analyze_payload()
        + ["---", ""]
        + analyze_csic()
        + ["---", ""]
        + build_recommendation()
    )
    report = "\n".join(sections) + "\n"

    NOTES_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[OK] EDA 결과를 {NOTES_PATH.relative_to(PROJECT_ROOT)} 에 저장했습니다.")
    print(f"[OK] 그림을 {FIG_DIR.relative_to(PROJECT_ROOT)} 에 저장했습니다.")


if __name__ == "__main__":
    main()
