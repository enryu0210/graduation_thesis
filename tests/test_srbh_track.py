"""합성 요청으로 매핑·감사·충돌 제거와 기존 CSV 형식 보존을 검증한다."""

import pandas as pd
import pytest

from src.data import preprocess, srbh_track


@pytest.mark.parametrize("name,expected", [
    ("F1", "uri"), ("F2", "uri\nbody"), ("F3", "uri\nbody\ncookie"),
    ("F4", "uri\nbody\ncookie\nua"), ("UC", "ua\ncookie"),
])
def test_compose_fields(name, expected):
    frame = pd.DataFrame(dict(request_http_request=["uri"], request_body=["body"],
                              request_cookie=["cookie"], request_user_agent=["ua"]))
    assert srbh_track.compose_fields(frame, name) == [expected]


def test_compose_empty_missing_and_nan():
    frame = pd.DataFrame(dict(request_http_request=["", None], request_body=[float("nan"), ""]))
    assert srbh_track.compose_fields(frame, "F2") == ["\n", "\n"]
    with pytest.raises(ValueError):
        srbh_track.compose_fields(frame, "F3")
    with pytest.raises(ValueError):
        srbh_track.compose_fields(frame, "unknown")


def test_field_rules_match_measurement():
    from src.analysis.measure_srbh_fields import build_combinations

    frame = make_raw([("000",), ("66",)])
    frame["request_referer"] = ""
    frame["label"] = ["Normal", "SQLInjection"]
    frame["row_id"] = [0, 1]
    frame["text_raw"] = frame.request_http_request + "\n" + frame.request_body
    measured = build_combinations(frame)
    for name in ("F1", "F2", "F3", "F4"):
        assert srbh_track.compose_fields(frame, name) == measured[name].tolist()


def make_raw(rows):
    codes = ["000", "66", "242", "88", "248", "126", "16", "34", "49", "100", "153", "272", "310", "549"]
    raw = pd.DataFrame([
        {f"{code} - 합성 라벨": "1" if code in active else "0" for code in codes}
        for active in rows
    ], columns=[f"{code} - 합성 라벨" for code in codes])
    raw["request_http_request"] = [f"/request/{i}" for i in range(len(raw))]
    raw["request_body"] = ""
    raw["request_cookie"] = "쿠키"
    raw["request_user_agent"] = "스캐너"
    return raw


def test_mapping_and_original_row_ids():
    raw = make_raw([("000",), ("88",), ("248",), ("242",), ("66", "88"), ("126",), ("88", "248"), ("66",), ("000",)])
    original = raw.copy(deep=True)
    frame, stats = srbh_track.build_srbh_frame(raw, {0})
    assert frame.row_id.tolist() == [1, 2, 3, 6, 7, 8]
    assert frame.label.tolist() == ["CommandInjection", "CommandInjection", "CodeInjection", "CommandInjection", "SQLInjection", "Normal"]
    assert stats == {
        "n_raw": 9, "ambiguous": 1, "non_target": 1,
        "after_mapping": {"total": 7, "by_class": dict(Normal=2, SQLInjection=1, CodeInjection=1, CommandInjection=3)},
        "e1a_removed": 1, "label_conflict_texts": 0, "label_conflict_rows": 0,
        "n_returned": {"total": 6, "by_class": dict(Normal=1, SQLInjection=1, CodeInjection=1, CommandInjection=3)},
    }
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize("row_id", [1, 2, 3, 99])
def test_audit_must_point_to_mapped_normal(row_id):
    raw = make_raw([("000",), ("66",), ("126",), ("000", "88")])
    with pytest.raises(ValueError, match="Normal"):
        srbh_track.build_srbh_frame(raw, {row_id})


def test_text_format_and_preserved_fields():
    raw = make_raw([("000",), ("66",)])
    raw.loc[0, "request_body"] = "body"
    frame, _ = srbh_track.build_srbh_frame(raw, set())
    assert frame.text_raw.tolist() == ["/request/0\nbody", "/request/1\n"]
    assert list(frame.columns) == srbh_track.OUTPUT_COLUMNS
    pd.testing.assert_frame_equal(frame[list(srbh_track.TEXT_FIELDS)], raw[list(srbh_track.TEXT_FIELDS)])


def test_all_conflicting_rows_removed_but_same_label_duplicates_retained():
    raw = make_raw([("000",), ("66",), ("000",), ("242",), ("242",)])
    raw["request_http_request"] = ["/conflict"] * 3 + ["/same"] * 2
    frame, stats = srbh_track.build_srbh_frame(raw, set())
    assert frame.row_id.tolist() == [3, 4]
    assert stats["label_conflict_texts"] == 1
    assert stats["label_conflict_rows"] == 3


def test_audit_removal_precedes_conflict_detection():
    raw = make_raw([("000",), ("66",)])
    raw["request_http_request"] = "/same"
    frame, stats = srbh_track.build_srbh_frame(raw, {0})
    assert frame.row_id.tolist() == [1]
    assert stats["label_conflict_rows"] == 0


def test_missing_audit_file_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(srbh_track, "AUDIT_CSV", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="python src/data/audit_srbh_labels.py"):
        srbh_track.load_srbh_4class()


@pytest.mark.parametrize("value", ["bad", "-1", "1.5", ""])
def test_invalid_audit_row_id(tmp_path, monkeypatch, value):
    audit = tmp_path / "audit.csv"
    pd.DataFrame({"row_id": [value]}).to_csv(audit, index=False)
    monkeypatch.setattr(srbh_track, "AUDIT_CSV", audit)
    with pytest.raises(ValueError, match="row_id"):
        srbh_track.load_srbh_4class()


def test_file_loader_preserves_literal_na(tmp_path, monkeypatch):
    raw = make_raw([("000",), ("66",)])
    raw.loc[1, "request_body"] = "NA"
    raw.to_csv(tmp_path / srbh_track.FILENAME, index=False)
    audit = tmp_path / "audit.csv"
    pd.DataFrame({"row_id": [0]}).to_csv(audit, index=False)
    monkeypatch.setattr(srbh_track, "RAW_DIR", tmp_path)
    monkeypatch.setattr(srbh_track, "AUDIT_CSV", audit)
    frame = srbh_track.load_srbh_4class()
    assert frame.row_id.tolist() == [1]
    assert frame.text_raw.tolist() == ["/request/1\nNA"]


@pytest.mark.parametrize("srbh", [False, True])
def test_save_track_columns(tmp_path, monkeypatch, srbh):
    monkeypatch.setattr(preprocess, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", tmp_path / "processed")
    columns = ["text_raw", "text_decoded", "label"]
    out = pd.DataFrame({"text_raw": ["/a\n"] * 3, "text_decoded": ["/a\n"] * 3,
                        "label": ["Normal"] * 3, "split": ["train", "val", "test"]})
    if srbh:
        out["row_id"] = [0, 1, 2]
        for field in srbh_track.TEXT_FIELDS:
            out[field] = ""
        columns += ["row_id", *srbh_track.TEXT_FIELDS]
    preprocess.save_track("synthetic", out)
    for split in ("train", "val", "test"):
        saved = pd.read_csv(tmp_path / "processed" / f"synthetic_{split}.csv", keep_default_na=False)
        assert list(saved.columns) == columns
        pd.testing.assert_frame_equal(saved, out.loc[out.split.eq(split), columns].reset_index(drop=True))


def test_empty_frame():
    frame, stats = srbh_track.build_srbh_frame(make_raw([]), set())
    assert frame.empty
    assert stats["n_returned"]["total"] == 0
