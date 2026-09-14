"""실제 요청 파일을 열지 않고 필드 측정의 경계와 봉인을 검증한다."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis import measure_srbh_fields as module


def make_frame(rows):
    records = []
    for i, values in enumerate(rows):
        record = {column: "" for column in module.FIELDS.values()}
        record.update(row_id=i, label="Normal")
        record.update({module.FIELDS.get(key, key): value for key, value in values.items()})
        record["text_raw"] = record["request_http_request"] + "\n" + record["request_body"]
        records.append(record)
    return pd.DataFrame(records)


def test_utf8_lengths():
    result = module.measure_lengths(make_frame([{"uri": "한"}]))
    assert result["by_class"]["overall"]["lengths"]["uri"] == dict(p50=3, p95=3, p99=3, max=3)


def test_combinations_preserve_empty_fields():
    frame = make_frame([{"uri": "/", "cookie": "c", "ua": "u"}, {}])
    before = frame.copy(deep=True)
    assert module.build_combinations(frame).to_dict("records") == [
        dict(F1="/", F2="/\n", F3="/\n\nc", F4="/\n\nc\nu"),
        dict(F1="", F2="\n", F3="\n\n", F4="\n\n\n"),
    ]
    pd.testing.assert_frame_equal(frame, before)


def test_mismatched_track_fails():
    frame = make_frame([{"uri": "/"}])
    frame.loc[0, "text_raw"] = "/"
    with pytest.raises(ValueError, match="F2 != text_raw"):
        module.build_combinations(frame)


def test_signal_locations_decode_and_partition():
    frame = make_frame([{"uri": "%27%20or%201%3D1"}, {"cookie": "SLEEP(5)"}, {"uri": "/hello"}])
    stats = module.measure_signals(frame)["by_class"]["Normal"]
    for key in ("in_f2", "outside_f2_only", "no_hit"):
        assert stats[key] == dict(count=1, rate=1 / 3)
    assert sum(stats[key]["count"] for key in ("in_f2", "outside_f2_only", "no_hit")) == stats["n_rows"]
    assert stats["outside_field_counts"] == dict(cookie=1, ua=0, referer=0)


def test_samples_are_sorted_limited_and_field_specific():
    frame = make_frame([{"cookie": "sleep(5)" + "x" * 250, "referer": "%3Cscript%3E"} for _ in range(7)])
    stats = module.measure_signals(frame.iloc[::-1])["by_class"]["Normal"]
    assert [sample["row_id"] for sample in stats["samples"]] == list(range(5))
    assert stats["outside_field_counts"] == dict(cookie=7, ua=0, referer=7)
    assert stats["samples"][0]["hit_fields"] == ["cookie", "referer"]
    assert len(stats["samples"][0]["decoded_prefixes"]["cookie"]) == 200
    assert stats["samples"][0]["decoded_prefixes"]["referer"] == "<script>"


@pytest.mark.parametrize("size, expected", [(2304, 0), (2305, 1)])
def test_byte_limit_is_strict(size, expected):
    stats = module.measure_lengths(make_frame([{"uri": "a" * size}]))["by_class"]["overall"]
    assert stats["exceeds"]["F1"] == dict(count=expected, rate=expected)


def test_sealed_split_not_in_source():
    assert "srbh_4class_test" not in Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("row_ids", [[0, 0], [-1, 1], [0.5, 1.0]])
def test_invalid_row_ids_fail(row_ids):
    frame = make_frame([{}, {}])
    frame["row_id"] = row_ids
    with pytest.raises(ValueError, match="row_id"):
        module.build_combinations(frame)


def test_shortcut_classifiers_use_train_and_evaluate_val():
    train = make_frame([{"uri": "aaaa", "ua": "agent-a", "label": "Normal"},
                        {"uri": "zzzz", "ua": "agent-z", "cookie": "z", "label": "SQLInjection"}] * 3)
    val = make_frame([{"uri": "aaaa", "ua": "agent-z", "label": "Normal"},
                      {"uri": "zzzz", "ua": "agent-z", "cookie": "z", "label": "SQLInjection"}])
    result = module.measure_shortcuts(train, val)
    assert result["proxy"] is True and result["status"] == "complete"
    assert result["most_common_train_ua"] == "agent-a"
    assert result["by_split"]["val"]["Normal"]["common_ua_rate"] == 0
    assert set(result["classifiers"]) == {"ua_cookie_only", "f2", "f4"}
    for outcome in result["classifiers"].values():
        assert {"macro_f1", "accuracy", "per_class", "roc_auc_ovr"} <= outcome["metrics"].keys()
        assert outcome["fit_seconds"] >= 0


def test_training_timeout_returns_partial_status(monkeypatch):
    frame = make_frame([{"uri": "a", "label": "Normal"},
                        {"uri": "z", "label": "SQLInjection"}])
    monkeypatch.setattr(module, "TRAINING_LIMIT_SECONDS", 0)
    result = module.measure_shortcuts(frame, frame)
    assert result["status"] == "timeout"
    assert result["interrupted_classifier"] == "ua_cookie_only"
    assert result["classifiers"] == {}
