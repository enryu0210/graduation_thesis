"""필드 선택이 대응표본과 test 봉인을 깨뜨리지 않는지 합성 파일로 검증한다."""

import json
import sys
import subprocess
import os

import numpy as np
import pandas as pd
import pytest

from src.eval import cross_validate as cv, cv_compare
import data_image
import data_text
from tagging import build_tag
from src.imaging.payload_to_image import payload_to_image, payload_to_rgb_image


@pytest.fixture
def pool_files(tmp_path, monkeypatch):
    monkeypatch.setattr(data_text, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(data_image, "IMAGES_DIR", tmp_path)
    for split in ("train", "val"):
        frame = pd.DataFrame(dict(request_http_request=[split, "001"], request_body=["", "NA"],
                                  request_cookie=["cookie", ""], request_user_agent=["ua", "ua"],
                                  label=["SQLInjection", "Normal"]))
        frame["text_raw"] = frame.request_http_request + "\n" + frame.request_body
        frame.to_csv(data_text.csv_path("srbh_4class", split), index=False)
        np.savez(data_image.npz_path("srbh_4class", split),
                 images=np.array([payload_to_image(t) for t in frame.text_raw]),
                 labels=[1, 0], classes=["Normal", "SQLInjection"])
    return tmp_path


@pytest.mark.parametrize("fields", [None, "F1", "F2", "F3", "F4", "UC"])
def test_sealed_pool_order_and_fingerprint(pool_files, fields):
    texts, y, classes = cv.load_text_pool("srbh_4class", "raw", fields=fields, exclude_test=True)
    images, image_y, image_classes = cv.load_image_pool("srbh_4class", "raw", 48, "gray", None,
                                                       exclude_test=True)
    assert len(texts) == len(images) == 4
    assert y.tolist() == [1, 0, 1, 0]
    assert classes == image_classes
    assert cv.label_fingerprint(y) == cv.label_fingerprint(image_y)
    if fields == "F2":
        assert texts == ["train\n", "001\nNA", "val\n", "001\nNA"]


@pytest.mark.parametrize("channels", ["gray", "rgb"])
def test_fields_images_never_load_npz(pool_files, monkeypatch, channels, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("필드 경로가 npz를 읽었습니다.")

    monkeypatch.setattr(data_image, "load_split", forbidden)
    encoders = ("raw_byte", "char_class", "byte_delta")
    images, y, _ = cv.load_image_pool("srbh_4class", "raw", 8, channels, encoders,
                                     fields="F3", exclude_test=True)
    texts, text_y, _ = cv.load_text_pool("srbh_4class", "raw", fields="F3", exclude_test=True)
    expected = (payload_to_rgb_image(texts[0], side=8, encoders=encoders)
                if channels == "rgb" else payload_to_image(texts[0], side=8))
    np.testing.assert_array_equal(images[0], expected)
    assert images.dtype == np.uint8
    assert cv.label_fingerprint(y) == cv.label_fingerprint(text_y)
    assert "메모리 이미지 변환:" in capsys.readouterr().out


def test_default_pool_includes_test(pool_files):
    path = data_text.csv_path("srbh_4class", "test")
    path.write_bytes(data_text.csv_path("srbh_4class", "val").read_bytes())
    assert len(cv.load_text_pool("srbh_4class", "raw")[0]) == 6


def test_image_import_in_fresh_process(pool_files):
    # 전체 테스트가 추가한 sys.path에 의존하면 실제 CLI에서만 import가 실패한다.
    script = (
        "from src.eval import cross_validate as cv; import data_text; from pathlib import Path; "
        "data_text.PROCESSED_DIR=Path(__import__('sys').argv[1]); "
        "cv.load_image_pool('srbh_4class','raw',8,'rgb',None,fields='F3',exclude_test=True)"
    )
    result = subprocess.run([sys.executable, "-c", script, str(pool_files)],
                            cwd=cv.PROJECT_ROOT, capture_output=True, text=True,
                            encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, result.stderr


def test_tag_defaults_and_suffixes():
    assert build_tag("payload_4class", "cnn", channels="rgb") == "payload_4class_cnn_raw_rgb"
    for fields in (None, "F2"):
        assert build_tag("srbh_4class", "cnn", fields=fields, sealed_test=False) == "srbh_4class_cnn_raw"
    assert build_tag("srbh_4class", "cnn", channels="rgb", fields="F3", sealed_test=True) == "srbh_4class_cnn_raw_rgb_fF3_sealed"


@pytest.mark.parametrize("args", [["--track", "payload_4class"], ["--track", "srbh_4class", "--text", "decoded"]])
def test_invalid_fields_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["cross_validate", "--model", "cnn", "--fields", "F3", *args])
    with pytest.raises(SystemExit) as error:
        cv.main()
    assert error.value.code == 2


def test_compare_pattern_and_figure(pool_files, monkeypatch):
    monkeypatch.setattr(cv_compare, "RESULTS_DIR", pool_files)
    monkeypatch.setattr(cv_compare, "PROJECT_ROOT", pool_files)
    monkeypatch.setattr(cv_compare, "FIG_DIR", pool_files / "figures")
    for track in ("payload_4class", "srbh_4class"):
        tag = f"{track}_cnn_raw_sealed"
        data = dict(tag=tag, folds=5, label_fingerprint=track,
                    summary={"macro_f1": {"per_fold": [.8, .81, .82, .83, .84]}})
        (pool_files / f"cv_{tag}.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="라벨 지문"):
        cv_compare.load_cv_results("macro_f1")
    monkeypatch.setattr(sys, "argv", ["cv_compare", "--pattern", "cv_srbh_4class_*_sealed.json",
                                      "--fig-name", "srbh_test"])
    cv_compare.main()
    assert (pool_files / "figures/srbh_test.png").is_file()
    assert cv_compare.short_name("srbh_4class_cnn_raw") == "cnn"
    assert cv_compare.short_name("payload_4class_cnn_raw") == "cnn"
