"""
Phase 8(RQ4b) — 네트워크 흐름(pcap) → 이미지 변환 핵심 모듈

목적:
    USTC-TFC2016 pcap 을 "세션(양방향 흐름)" 단위로 묶고, 각 세션의 앞 N바이트를
    RQ1 과 동일한 payload_to_image(48×48) 규격으로 이미지화한다.
    → "내용(payload)이 암호화로 사라져도 흐름 수준 신호는 남는다"를 같은 이미징 패러다임으로 검증(RQ4b).

핵심 설계 결정:
    1) 세션 정의: 5-tuple(proto, {ip:port}_a, {ip:port}_b)의 양방향 묶음.
       한 TCP/UDP 대화의 양방향 패킷을 한 흐름으로 본다(Wang et al. "session" 방식).
    2) 전계층(all-layer) 바이트 사용: 이더넷~L7 원본 바이트를 캡처 순서로 이어 붙인다.
       패킷 크기·프로토콜 필드·초기 핸드셰이크 등 side-channel 신호를 최대한 담기 위함.
    3) ⚠️ 주소 무력화(sanitization): 이더넷 MAC(src/dst)과 IP 주소(src/dst)를 0으로 덮는다.
       USTC-TFC2016 은 특정 IP/MAC 이 클래스와 강하게 결합되어 있어, 이를 지우지 않으면
       CNN 이 "트래픽 구조"가 아니라 "주소를 암기"해 부정 성능을 낸다(data-snooping).
       Wang et al. 도 동일한 sanitization 을 수행 → 정직한 비교를 위해 재현.

주의:
    - 이 모듈은 순수 변환(파일 I/O 최소)만 담당. 데이터셋 조립·분할·저장은 build_flow_dataset.py.
    - dpkt 로 classic pcap 을 읽는다(USTC 파일은 .pcap classic). pcapng 는 예외 처리로 감지.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import dpkt

from payload_to_image import DEFAULT_SIDE, bytes_to_image, bytes_to_rgb_image
from channel_encoders import DEFAULT_RGB_ENCODERS

# 세션당 사용할 최대 바이트 수 = 48*48 이미지 용량(2304B). 페이로드 이미지와 동일 규격.
DEFAULT_MAX_BYTES = DEFAULT_SIDE * DEFAULT_SIDE


def _open_pcap_reader(f):
    """classic pcap / pcapng 를 모두 읽을 수 있게 리더를 고른다."""
    try:
        return dpkt.pcap.Reader(f)
    except ValueError:
        f.seek(0)
        return dpkt.pcapng.Reader(f)


def _sanitize_and_serialize(eth: "dpkt.ethernet.Ethernet") -> bytes:
    """이더넷 프레임의 MAC/IP 주소를 0으로 덮고 다시 바이트로 직렬화한다.

    왜: 주소가 클래스 라벨과 결합되어 있으면 CNN 이 주소를 암기해 부정 성능을 낸다.
    포트/페이로드/패킷크기 등 '흐름의 행동' 신호만 남기기 위해 식별자를 제거한다.
    """
    eth.src = b"\x00" * 6
    eth.dst = b"\x00" * 6
    inner = eth.data
    # IPv4/IPv6 주소 제거(있을 때만). ARP 등은 그대로 둔다(식별자 결합 위험이 낮음).
    if isinstance(inner, dpkt.ip.IP):
        inner.src = b"\x00" * 4
        inner.dst = b"\x00" * 4
    elif isinstance(inner, dpkt.ip6.IP6):
        inner.src = b"\x00" * 16
        inner.dst = b"\x00" * 16
    return bytes(eth)


def _session_key(eth: "dpkt.ethernet.Ethernet"):
    """양방향 세션 키를 만든다. TCP/UDP 가 아니면 None(해당 패킷은 세션 대상 제외)."""
    inner = eth.data
    if not isinstance(inner, (dpkt.ip.IP, dpkt.ip6.IP6)):
        return None
    l4 = inner.data
    if not isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        return None
    proto = inner.p if isinstance(inner, dpkt.ip.IP) else inner.nxt
    endpoint_a = (bytes(inner.src), l4.sport)
    endpoint_b = (bytes(inner.dst), l4.dport)
    # frozenset 으로 방향(A→B / B→A)을 하나의 세션으로 묶는다.
    return (proto, frozenset((endpoint_a, endpoint_b)))


def iter_session_bytes(
    pcap_path: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_sessions: int | None = None,
) -> Iterator[bytes]:
    """pcap 을 세션 단위로 묶어, 각 세션의 (sanitize 된) 앞 max_bytes 바이트열을 순서대로 내보낸다.

    - 각 세션은 앞 max_bytes 까지만 누적(뒤는 버림) → 메모리 절약 + 이미지 용량과 일치.
    - max_sessions: 클래스별 표본 상한(불균형·대용량 pcap 대비). None 이면 제한 없음.
    반환 순서: 세션이 max_bytes 를 채우는 즉시(또는 파일 끝에서) 확정해 내보낸다.
    """
    sessions: dict = {}      # key → bytearray(누적 바이트)
    done: set = set()        # 이미 내보낸(=가득 찬) 세션 키
    emitted = 0

    with open(pcap_path, "rb") as f:
        reader = _open_pcap_reader(f)
        for _ts, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.dpkt.UnpackError, Exception):
                continue  # 파싱 불가 프레임은 건너뛴다(견고성)

            key = _session_key(eth)
            if key is None or key in done:
                continue

            try:
                data = _sanitize_and_serialize(eth)
            except Exception:
                continue

            buf_acc = sessions.setdefault(key, bytearray())
            need = max_bytes - len(buf_acc)
            if need > 0:
                buf_acc.extend(data[:need])

            # 세션이 max_bytes 를 채우면 즉시 확정·방출(스트리밍, 메모리 상한 유지).
            if len(buf_acc) >= max_bytes:
                yield bytes(buf_acc)
                done.add(key)
                del sessions[key]
                emitted += 1
                if max_sessions and emitted >= max_sessions:
                    return

    # 파일 끝: max_bytes 를 못 채운 짧은 세션들도 (zero-padding 되어) 방출한다.
    for key, buf_acc in sessions.items():
        if max_sessions and emitted >= max_sessions:
            break
        yield bytes(buf_acc)
        emitted += 1


def session_to_image(session_bytes: bytes, side: int = DEFAULT_SIDE):
    """세션 바이트열을 (side, side) 그레이스케일 이미지로(RQ1 규격 재사용)."""
    return bytes_to_image(session_bytes, side=side)


def session_to_rgb_image(session_bytes: bytes, side: int = DEFAULT_SIDE,
                         encoders: tuple[str, str, str] = DEFAULT_RGB_ENCODERS):
    """세션 바이트열을 (side, side, 3) RGB 이미지로(payload 와 동일 채널 인코더 재사용)."""
    return bytes_to_rgb_image(session_bytes, side=side, encoders=encoders)
