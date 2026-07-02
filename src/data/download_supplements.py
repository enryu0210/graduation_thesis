"""
페이로드 다양성 보강 소스 다운로드 스크립트 (Phase 1)

목적:
    특정 데이터셋 패턴에 과적합되는 것을 막기 위한 보조 페이로드 소스를 확보한다.
    계획서(01_data_acquisition_plan.md 2.3 / 3.3)에 명시된 GitHub 리스트들을
    git clone 으로 받고, 재현성을 위해 "고정된 커밋 해시"를 기록한다.

왜 커밋 해시를 기록하는가:
    이 저장소들은 계속 갱신되므로, 몇 개월 뒤 다시 clone 하면 내용이 달라진다.
    논문 재현성을 위해 "어느 시점의 스냅샷을 썼는지"를 커밋 해시로 못박아 둔다.

사용법:
    python src/data/download_supplements.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = PROJECT_ROOT / "data" / "raw" / "payload_supplement"

# (폴더명, git URL) — 계획서 2.3 표의 소스들
REPOS = [
    ("PayloadsAllTheThings", "https://github.com/swisskyrepo/PayloadsAllTheThings.git"),
    ("sql-injection-payload-list", "https://github.com/payloadbox/sql-injection-payload-list.git"),
    ("command-injection-payload-list", "https://github.com/payloadbox/command-injection-payload-list.git"),
]


def clone_or_report(name: str, url: str) -> tuple[str, str]:
    """저장소를 clone(또는 이미 있으면 skip)하고 현재 커밋 해시를 반환한다.

    반환: (폴더명, 커밋해시 or 오류메시지)
    """
    target = DEST_DIR / name

    if target.exists():
        print(f"[skip] 이미 존재: {name}")
    else:
        print(f"[clone] {url}")
        try:
            # --depth 1: 전체 히스토리는 불필요하므로 최신 스냅샷만 받아 용량을 아낀다.
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # 실패를 조용히 넘기지 않고 stderr 를 그대로 보고한다.
            return name, f"clone 실패: {exc.stderr.strip()}"

    # 현재 커밋 해시 조회 (재현성 기록용)
    try:
        result = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return name, result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return name, f"해시 조회 실패: {exc.stderr.strip()}"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 페이로드 보강 소스 다운로드 ===")
    results = [clone_or_report(name, url) for name, url in REPOS]

    print("\n=== 커밋 해시 (DATA_MANIFEST.md 에 기록하세요) ===")
    for name, info in results:
        print(f"  - {name}: {info}")


if __name__ == "__main__":
    main()
