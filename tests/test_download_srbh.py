"""작은 로컬 파일만 사용하여 원본 무결성 판정의 실패 조건을 확인한다."""

import hashlib
import io
import urllib.error

import pytest

from src.data.download_srbh import file_checksums, validate_integrity
from src.data import download_srbh


@pytest.fixture
def small_file(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(b"name,label\nhello,1\n")
    return path


def test_matching_md5(small_file):
    size, md5, sha256 = file_checksums(small_file)
    validate_integrity(size, md5, 19, "7c2895e88f80551a17c7462182e0e44f")
    assert sha256 == hashlib.sha256(small_file.read_bytes()).hexdigest()


def test_mismatching_md5(small_file):
    size, md5, _ = file_checksums(small_file)
    with pytest.raises(ValueError, match="MD5 불일치"):
        validate_integrity(size, md5, size, "0" * 32)


def test_mismatching_size(small_file):
    size, md5, _ = file_checksums(small_file)
    with pytest.raises(ValueError, match="크기 불일치"):
        validate_integrity(size, md5, size + 1, md5)


@pytest.fixture
def local_download(monkeypatch, tmp_path):
    content = b"name,label\nhello,1\n"
    monkeypatch.setattr(download_srbh, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download_srbh, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(download_srbh, "EXPECTED_SIZE", len(content))
    monkeypatch.setattr(download_srbh, "EXPECTED_MD5", hashlib.md5(content).hexdigest())
    return tmp_path / download_srbh.FILENAME, content


def test_verified_file_skips_network(local_download, monkeypatch, capsys):
    destination, content = local_download
    destination.write_bytes(content)

    def unexpected_request(*args, **kwargs):
        pytest.fail("검증된 파일이 있으므로 네트워크를 호출하면 안 됩니다.")

    monkeypatch.setattr(download_srbh.urllib.request, "urlopen", unexpected_request)
    download_srbh.download()
    assert "이미 존재·검증 통과" in capsys.readouterr().out


@pytest.mark.parametrize("valid", [True, False])
def test_force_replaces_only_verified_file(local_download, monkeypatch, valid):
    destination, content = local_download
    destination.write_bytes(content)
    # 동일 크기의 손상을 써야 MD5 검사까지 도달하는 실패 경로를 검증할 수 있다.
    response = io.BytesIO(content if valid else b"x" * len(content))
    response.headers = {"Content-Type": "text/csv"}
    calls = []

    def open_local_response(*args, **kwargs):
        calls.append(True)
        return response

    monkeypatch.setattr(download_srbh.urllib.request, "urlopen", open_local_response)
    if valid:
        download_srbh.download(force=True)
    else:
        with pytest.raises(ValueError, match="MD5 불일치"):
            download_srbh.download(force=True)
    assert calls == [True]
    assert destination.read_bytes() == content
    assert not destination.with_suffix(".csv.part").exists()


def test_network_error_removes_partial(local_download, monkeypatch):
    destination, _ = local_download
    temporary = destination.with_suffix(".csv.part")
    temporary.write_bytes(b"interrupted")

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("테스트용 연결 실패")

    monkeypatch.setattr(download_srbh.urllib.request, "urlopen", unavailable)
    with pytest.raises(urllib.error.URLError):
        download_srbh.download()
    assert not temporary.exists()
    assert not destination.exists()
