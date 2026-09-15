"""감사 중복과 바이트 차이를 보존하는지 파일 없이 검증한다."""

import pandas as pd
import pytest

from src.analysis.build_srbh_e1 import build_e1a_set, detection_report
from src.eval.metrics import wilson_ci


def test_overlap_and_outside_preserve_rows():
    raw = pd.DataFrame({"request_http_request": ["a", "a", " a"],
                        "request_body": ["", "", ""], "request_cookie": ["x", "y", "z"]})
    flags = pd.DataFrame({"row_id": [0, 1, 2], "audit_classes": ["SQLInjection"] * 3,
                          "patterns": ["p"] * 3,
                          "fields": ["request_cookie", "request_cookie;request_body", "request_http_request"]})
    track = {"train": pd.DataFrame({"text_raw": ["a\n"]}),
             "val": pd.DataFrame({"text_raw": ["a\n"]}),
             "test": pd.DataFrame({"text_raw": []})}
    frame, stats = build_e1a_set(raw, flags, track)
    assert frame.row_id.tolist() == [0, 1, 2]
    assert frame.outside_f2.tolist() == [True, False, False]
    assert frame.overlap_split.tolist() == ["train;val", "train;val", ""]
    assert stats["n_unique_text_f2"] == 2
    assert stats["n_unique_text_f3"] == 3
    assert stats["overlap_with_track"]["train"] == {"unique_texts": 1, "rows": 2}
    flags.loc[0, "row_id"] = 99
    with pytest.raises(ValueError, match="원본"):
        build_e1a_set(raw, flags, track)


@pytest.mark.parametrize("pred", [[], ["Normal"], ["SQLInjection", "Normal", "CodeInjection"]])
def test_detection_uses_shared_interval(pred):
    report = detection_report(pred, confidence=0.9)
    detected = sum(label != "Normal" for label in pred)
    assert report["n"] == len(pred)
    assert report["detection_rate"] == wilson_ci(detected, len(pred), 0.9)
    assert report["benign_evasion"] == wilson_ci(len(pred) - detected, len(pred), 0.9)


def test_empty_audit_keeps_schema():
    raw = pd.DataFrame(columns=["request_http_request", "request_body", "request_cookie"])
    flags = pd.DataFrame(columns=["row_id", "audit_classes", "patterns", "fields"])
    frame, stats = build_e1a_set(raw, flags, {"test": pd.DataFrame({"text_raw": []})})
    assert frame.empty and "overlap_split" in frame
    assert stats["n_rows"] == 0
