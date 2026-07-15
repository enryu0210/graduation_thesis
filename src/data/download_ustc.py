"""
Phase 8(RQ4b) — USTC-TFC2016 암호화/악성 트래픽 데이터셋 다운로드·해제 스크립트

목적:
    RQ4b(신호의 이동: 흐름 side-channel 이미지 탐지)용 실측 데이터를 확보한다.
    GitHub 저장소(davidyslu/USTC-TFC2016, MPL-2.0)에서 20개 트래픽 파일(악성 10 + 정상 10)을
    내려받아 data/raw/ustc_tfc2016/{Benign,Malware}/ 아래에 pcap 으로 풀어 둔다.

배경(왜 이 데이터인가):
    docs/06_encryption_boundary_rq4_design.md 참조. 내용(payload)이 암호화로 사라져도
    흐름 수준 신호(패킷 크기 시퀀스·바이트 분포)는 남는다 → "같은 이미지화 패러다임"으로
    악성/정상 흐름을 탐지할 수 있는지 실측. Wang et al.(2017) 이 만든 정본 데이터셋.

라이선스/출처(논문 4장에 반드시 명시):
    - Dataset: USTC-TFC2016, Wei Wang et al. / echowei-DeepTraffic, MPL-2.0.
    - 저장소는 일부 파일을 .7z 로 압축 보관(총 압축 ~387MB, 해제 시 ~3.71GB).

사용법:
    python src/data/download_ustc.py               # 전체 20종 다운로드+해제
    python src/data/download_ustc.py --keep-archives  # .7z 원본도 보존
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import py7zr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ustc_tfc2016"
RAW_BASE_URL = "https://raw.githubusercontent.com/davidyslu/USTC-TFC2016/master"

# (카테고리, 저장소상 파일명). .7z 는 다운로드 후 해제한다.
FILES: list[tuple[str, str]] = [
    ("Benign", "BitTorrent.pcap"), ("Benign", "FTP.pcap"), ("Benign", "Facetime.pcap"),
    ("Benign", "Gmail.pcap"), ("Benign", "MySQL.pcap"), ("Benign", "Outlook.pcap"),
    ("Benign", "SMB.7z"), ("Benign", "Skype.pcap"), ("Benign", "Weibo.7z"),
    ("Benign", "WorldOfWarcraft.pcap"),
    ("Malware", "Cridex.7z"), ("Malware", "Geodo.7z"), ("Malware", "Htbot.7z"),
    ("Malware", "Miuref.pcap"), ("Malware", "Neris.7z"), ("Malware", "Nsis-ay.7z"),
    ("Malware", "Shifu.7z"), ("Malware", "Tinba.pcap"), ("Malware", "Virut.7z"),
    ("Malware", "Zeus.pcap"),
]


def sha256_of(path: Path) -> str:
    """파일 SHA256(재현성·DATA_MANIFEST 기록용)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """스트리밍 다운로드(대용량 대비 메모리 절약). 이미 있으면 건너뛴다."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    [skip] 이미 존재: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # 일부 CDN 이 기본 UA 를 거부할 수 있어 UA 를 명시한다.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (thesis-fetch)"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    받는 중 {dest.name}: {got/1e6:6.1f}/{total/1e6:.1f} MB", end="")
    print()
    tmp.rename(dest)  # 완전히 받은 뒤에만 최종 이름으로(중단 시 .part 로 남아 재시도 안전)


def extract_7z(archive: Path, out_dir: Path) -> list[Path]:
    """.7z 를 해제하고 생성된 파일 경로 목록을 반환한다."""
    before = set(out_dir.glob("*"))
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)
    after = set(out_dir.glob("*"))
    return sorted(after - before)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="USTC-TFC2016 다운로드·해제 (RQ4b)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="해제 후 .7z 원본을 삭제하지 않고 보존")
    args = parser.parse_args()

    print(f"=== USTC-TFC2016 다운로드 → {RAW_DIR.relative_to(PROJECT_ROOT)} ===")
    pcaps: list[Path] = []
    for category, filename in FILES:
        url = f"{RAW_BASE_URL}/{category}/{filename}"
        dest = RAW_DIR / category / filename
        try:
            download(url, dest)
        except Exception as exc:  # 네트워크 오류 등: 한 파일 실패가 전체를 막지 않도록
            print(f"    [실패] {category}/{filename}: {exc}")
            continue

        if dest.suffix == ".7z":
            try:
                produced = extract_7z(dest, dest.parent)
                pcaps.extend(p for p in produced if p.suffix == ".pcap")
                print(f"    [해제] {filename} → {[p.name for p in produced]}")
                if not args.keep_archives:
                    dest.unlink()  # 공간 절약(원본은 언제든 재다운로드 가능)
            except Exception as exc:
                print(f"    [해제 실패] {filename}: {exc}")
        else:
            pcaps.append(dest)

    print(f"\n[완료] pcap {len(pcaps)}개 확보. 다음: build_flow_dataset.py 로 이미지화.")


if __name__ == "__main__":
    main()
