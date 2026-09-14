"""합성 요청만으로 디코딩, 요청 단위 집계와 원본 행 번호 보존을 검증한다."""

import re

import pandas as pd
import pytest

from src.data import audit_srbh_labels as audit


def make_frame(texts, index=None):
    frame = pd.DataFrame({field: [""] * len(texts) for field in audit.SCAN_FIELDS}, index=index)
    frame[audit.SCAN_FIELDS[0]] = texts
    codes = ("000", "66", "242", "88", "248", "126", "16", "34", "49", "100", "153", "272", "310", "549")
    for code in codes:
        frame[f"{code} - 합성 라벨"] = "1" if code == "000" else "0"
    return frame


@pytest.mark.parametrize("text", ["%3Cscript%3E", "%25253CSCRIPT%25253E", "&lt;script&gt;"])
def test_decoding_before_scan(text):
    assert audit.scan_frame(make_frame([text])).iloc[0].patterns == "script_tag"


def test_multiple_classes_and_input_index_order():
    frame = make_frame(["UNION SELECT x;cat /etc/passwd", "무해", "<script>"], [31, 7, 2])
    original = frame.copy(deep=True)
    flags = audit.scan_frame(frame)
    assert flags.row_id.tolist() == [31, 2]
    assert flags.iloc[0].audit_classes == "CmdI;SQLi"
    pd.testing.assert_frame_equal(frame, original)


def test_benign_paths_and_browser_agents():
    frame = make_frame(["/wp-content/themes/a.css"] * 3)
    frame.request_user_agent = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
    ]
    assert audit.scan_frame(frame).empty


def test_excerpt_and_deterministic_request_samples():
    text = "a" * 110 + "%3Cscript%3E" + "b" * 120
    frame = make_frame([text] * 12, list(range(11, -1, -1)))
    frame.request_referer = frame.request_http_request
    result, flags = audit.audit_frame(frame)
    samples = result["samples"]["script_tag"]
    assert [item["row_id"] for item in samples] == list(range(10))
    assert samples[0]["matches"] == [
        {"field": field, "snippet": "a" * 80 + "<script" + ">" + "b" * 79}
        for field in ("request_http_request", "request_referer")
    ]
    normal = result["normal"]
    assert normal["pattern_counts"]["script_tag"] == 12
    assert normal["field_pattern_counts"]["request_referer"]["script_tag"] == 12
    assert normal["field_counts"]["request_http_request"] == 12
    assert normal["class_combinations"] == {"CodeInj": 12}
    assert flags.row_id.tolist() == list(range(12))


def test_excerpt_at_boundaries_and_no_match():
    pattern = re.compile("<script", re.IGNORECASE)
    assert audit.match_excerpt("<script>", pattern) == "<script>"
    with pytest.raises(ValueError, match="적중"):
        audit.match_excerpt("무해", pattern)


def test_exact_signature_names():
    assert set(audit.SIGNATURES_V1) == {
        "union_select", "tautology", "time_based", "schema_probe", "error_based",
        "stacked_query", "comment_terminator", "script_tag", "event_handler", "js_uri",
        "dangerous_tag", "js_sink", "php_code", "shell_chain", "subshell", "shell_path",
    }


def test_reference_mapping_and_normal_only_flags():
    frame = make_frame(["<script>"] * 8)
    frame.loc[1:, "000 - 합성 라벨"] = "0"
    for row, codes in enumerate([(), ("66",), ("242",), ("88",), ("248",), ("88", "248"), ("66", "88"), ("126",)]):
        for code in codes:
            frame.loc[row, f"{code} - 합성 라벨"] = "1"
    result, flags = audit.audit_frame(frame)
    assert flags.row_id.tolist() == [0]
    assert {label: value["n_scanned"] for label, value in result["attack_reference"].items()} == {"SQLi": 1, "CodeInj": 1, "CmdI": 3}
    assert all(value["flag_rate"] == 1 for value in result["attack_reference"].values())


def test_empty_frame():
    result, flags = audit.audit_frame(make_frame([]))
    assert flags.empty
    assert list(flags.columns) == ["row_id", "audit_classes", "patterns", "fields"]
    assert result["normal"]["flag_rate"] is None


@pytest.mark.parametrize("change", ["field", "index", "label"])
def test_invalid_input_rejected(change):
    frame = make_frame(["", ""])
    if change == "field":
        frame = frame.drop(columns="request_referer")
    elif change == "index":
        frame.index = [0, 0]
    else:
        frame.loc[0, "66 - 합성 라벨"] = "잘못된 값"
    with pytest.raises(ValueError):
        audit.audit_frame(frame)


def test_missing_file_guidance(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit, "RAW_DIR", tmp_path)
    monkeypatch.setattr(audit.sys, "argv", ["audit_srbh_labels.py"])
    assert audit.main() == 1
    assert "python src/data/download_srbh.py" in capsys.readouterr().err


def test_only_decoded_string_is_scanned():
    # 디코딩이 원문의 '+'를 공백으로 바꾸므로 raw 전용 적중은 집계하지 않아야 한다.
    custom = {"literal_plus": {"class": "SQLi", "pattern": r"a\+b"}}
    assert audit.scan_frame(make_frame(["a+b"]), custom).empty
