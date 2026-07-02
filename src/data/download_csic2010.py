"""
CSIC 2010 HTTP dataset 다운로드 스크립트 (Phase 1)

목적:
    웹 트래픽(HTTP 요청) 데이터인 CSIC 2010 을 확보한다.
    계획서(01_data_acquisition_plan.md 3.2)대로 Kaggle 미러를 1차 소스로 쓰고,
    가능하면 GitLab 미러와 행 수를 교차검증한다.

왜 Kaggle 미러인가:
    공식 원본(isi.csic.es)은 접속 불안정 리포트가 많아 재현이 어렵다.
    Kaggle 미러(ispangler/...)는 안정적으로 접근 가능하다.

사전 준비 (중요):
    Kaggle 다운로드에는 인증이 필요하다. 아래 중 하나가 준비돼 있어야 한다.
      1) ~/.kaggle/kaggle.json  (Kaggle 계정 > Settings > API > Create New Token)
      2) 환경변수 KAGGLE_USERNAME / KAGGLE_KEY
    인증이 없으면 이 스크립트는 방법을 안내하고 종료한다 (임의로 진행하지 않음).

사용법:
    python src/data/download_csic2010.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = PROJECT_ROOT / "data" / "raw" / "csic2010"

KAGGLE_DATASET = "ispangler/csic-2010-web-application-attacks"


def has_kaggle_credentials() -> bool:
    """Kaggle 인증 정보가 있는지 확인한다."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def print_credential_help() -> None:
    """인증이 없을 때 사용자가 직접 설정할 수 있도록 안내한다."""
    print(
        "\n[중단] Kaggle 인증 정보를 찾지 못했습니다.\n"
        "다음 중 한 가지로 인증을 설정한 뒤 다시 실행하세요:\n"
        "  1) https://www.kaggle.com/settings 에서 'Create New Token' 을 눌러\n"
        "     내려받은 kaggle.json 을 다음 위치에 두기:\n"
        f"       {Path.home() / '.kaggle' / 'kaggle.json'}\n"
        "  2) 또는 환경변수 설정: KAGGLE_USERNAME, KAGGLE_KEY\n"
    )


def main() -> None:
    if not has_kaggle_credentials():
        print_credential_help()
        return

    # kagglehub 는 인증이 준비돼 있으면 캐시 경로에 데이터를 내려받아 준다.
    try:
        import kagglehub
    except ImportError:
        print("[오류] kagglehub 이 설치돼 있지 않습니다. `pip install kagglehub` 후 재시도하세요.")
        return

    print(f"[진행] Kaggle 미러 다운로드: {KAGGLE_DATASET}")
    try:
        cached_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        # 네트워크/권한 오류를 삼키지 않고 명확히 보고한다.
        print(f"[오류] 다운로드 실패: {exc}")
        return

    # 캐시 경로의 파일을 프로젝트 data/raw/csic2010 로 복사(원본 보존 원칙 유지).
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in cached_path.rglob("*"):
        if src.is_file():
            dst = DEST_DIR / src.name
            shutil.copy2(src, dst)
            copied += 1

    print(f"[완료] {copied}개 파일을 {DEST_DIR.relative_to(PROJECT_ROOT)} 로 복사했습니다.")
    print(
        "\n[다음 할 일]\n"
        "  - GitLab 미러(gitlab.fing.edu.uy/gsi/web-application-attacks-datasets)와\n"
        "    행 수/정상·이상 비율을 교차검증하고 DATA_MANIFEST.md 에 결과를 기록하세요.\n"
        "  - 공식 통계(정상 36,000 + 이상 25,000+)와 실제 수치를 대조하세요."
    )


if __name__ == "__main__":
    main()
