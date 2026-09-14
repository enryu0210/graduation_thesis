"""
Phase 13 — SR-BH 2020 웹 요청 원본 다운로드 스크립트

목적:
    새 트랙 설계에 앞서 원본 CSV를 변환 없이 재현 가능하게 확보한다.

라이선스/출처:
    - SR-BH 2020, Harvard Dataverse, CC0 1.0.
    - https://doi.org/10.7910/DVN/OGOIXX (버전 1.2, 2022-06-06)
    - dataFile 6319496, data_capec_multilabel.csv.

사용법:
    python src/data/download_srbh.py          # 검증된 기존 파일은 재사용
    python src/data/download_srbh.py --force  # 원본을 강제로 다시 다운로드
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "srbh2020"
FILENAME = "data_capec_multilabel.csv"
DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/6319496"
EXPECTED_SIZE = 436_437_661
EXPECTED_MD5 = "173ec515308bdce5aec19cfd5b792596"
CHUNK_SIZE = 1 << 20


def validate_integrity(size: int, md5: str, expected_size: int, expected_md5: str) -> None:
    """입력값만 비교하여 네트워크·파일 변경 없이 무결성을 판정한다."""
    if size != expected_size:
        raise ValueError(f"크기 불일치: 실제 {size} bytes / 기대 {expected_size} bytes")
    if md5.lower() != expected_md5.lower():
        raise ValueError(f"MD5 불일치: 실제 {md5} / 기대 {expected_md5}")


def file_checksums(path: Path) -> tuple[int, str, str]:
    """큰 원본을 메모리에 올리지 않고 한 번 읽어 두 해시를 계산한다."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def verify_file(path: Path) -> None:
    """정본으로 인정하기 전에 고정된 버전의 크기와 MD5를 모두 확인한다."""
    size, md5, sha256 = file_checksums(path)
    validate_integrity(size, md5, EXPECTED_SIZE, EXPECTED_MD5)
    print(f"[검증 통과] 크기: {size} bytes\nMD5: {md5}\nSHA256: {sha256}", flush=True)


def download(force: bool = False) -> None:
    """검증 완료 전에는 정본을 교체하지 않아 실패 시 기존 원본을 보존한다."""
    destination = RAW_DIR / FILENAME
    if destination.exists() and not force:
        try:
            verify_file(destination)
        except ValueError as exc:
            print(f"[재다운로드] 기존 파일 검증 실패: {exc}", flush=True)
        else:
            print("[건너뜀] 이미 존재·검증 통과 — 다운로드하지 않음", flush=True)
            return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0 (thesis-fetch)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            print(f"[응답] Content-Type: {response.headers.get('Content-Type')}", flush=True)
            received = 0
            next_progress = 0
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                received += len(chunk)
                if received >= next_progress:
                    print(f"[다운로드] {received}/{EXPECTED_SIZE} bytes ({received / EXPECTED_SIZE:.1%})", flush=True)
                    next_progress = received + 32 * CHUNK_SIZE
        verify_file(temporary)
        temporary.replace(destination)
        print(f"[완료] {destination.relative_to(PROJECT_ROOT)}", flush=True)
    finally:
        # 검증 실패나 사용자의 중단 후 불완전한 임시 파일을 재사용하지 않는다.
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="SR-BH 2020 원본 다운로드·무결성 검증")
    parser.add_argument("--force", action="store_true", help="기존 파일이 검증되어도 다시 다운로드")
    args = parser.parse_args()
    try:
        download(force=args.force)
    except KeyboardInterrupt:
        print("[중단] 사용자가 다운로드를 중단했습니다.", file=sys.stderr, flush=True)
        return 130
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
        print(f"[실패] 다운로드 또는 원본 검증에 실패했습니다: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
