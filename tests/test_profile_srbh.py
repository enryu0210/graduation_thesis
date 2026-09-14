"""손으로 대조 가능한 합성 요청으로 라벨 집계와 매핑 경계를 검증한다."""

import pandas as pd
import pytest

from src.data import profile_srbh
from src.data.profile_srbh import TEXT_FIELDS, detect_label_columns, profile_labels


@pytest.fixture
def sample():
    codes = ["000", "66", "242", "88", "248", "126", "16", "34", "49", "100", "153", "272", "310", "549"]
    labels = [f"{code} - 합성 라벨" for code in codes]
    rows = [("000",), ("66",), ("242",), ("88",), ("248",), ("88", "248"), ("126",), ("66", "88"), ("000", "248"), ()]
    frame = pd.DataFrame([{name: "1" if code in row else "0" for code, name in zip(codes, labels)} for row in rows])
    for name in TEXT_FIELDS:
        frame[name] = ["", "NA"] * 5
    return frame


@pytest.mark.parametrize("change", ["drop", "add"])
def test_label_count_must_be_fourteen(sample, change):
    if change == "drop":
        sample = sample.drop(columns=sample.columns[0])
    else:
        sample["999 - 추가 라벨"] = "0"
    with pytest.raises(ValueError, match="14개"):
        detect_label_columns(sample)


def test_command_structure(sample):
    structure = profile_labels(sample)["cmd_injection_structure"]
    assert (structure["only_88"], structure["only_248"], structure["both_88_248"]) == (2, 2, 1)
    assert structure["p_248_given_88"] == pytest.approx(1 / 3)
    assert structure["p_88_given_248"] == pytest.approx(1 / 3)
    assert structure["only_248_other_label_counts"]["000 - 합성 라벨"] == 1
    assert sum(structure["only_248_other_label_counts"].values()) == 1


def test_248_only_changes_scenario(sample):
    scenarios = profile_labels(sample.iloc[[4]])["mapping_scenarios"]
    assert scenarios["A_merge_248"]["CmdI"] == 1
    assert scenarios["B_drop_248_only"]["excluded"] == 1


def test_nontarget_is_excluded(sample):
    for counts in profile_labels(sample.iloc[[6]])["mapping_scenarios"].values():
        assert counts["excluded"] == 1
        assert counts["Normal"] == 0


def test_sqli_and_command_are_ambiguous(sample):
    for counts in profile_labels(sample.iloc[[7]])["mapping_scenarios"].values():
        assert counts["ambiguous"] == 1


def test_invalid_values_are_reported(sample):
    sample.loc[0, "66 - 합성 라벨"] = "bad"
    sample.loc[1, "66 - 합성 라벨"] = ""
    result = profile_labels(sample)
    assert result["invalid_label_values"]["n_cells"] == 2
    assert result["invalid_label_values"]["n_rows"] == 2
    assert result["invalid_label_values"]["by_label"]["66 - 합성 라벨"] == {"bad": 1, "": 1}
    assert result["label_counts"]["66 - 합성 라벨"] == 1


def test_complete_counts_and_consistency(sample):
    original = sample.copy(deep=True)
    result = profile_labels(sample)
    assert result["label_columns"] == list(sample.columns[:14])
    assert result["cardinality"] == {"0": 1, "1": 6, "2": 3}
    assert result["normal_conflict"] == 1
    assert result["mapping_scenarios"] == {
        "A_merge_248": dict(Normal=1, SQLi=1, CodeInj=1, CmdI=3, ambiguous=2, excluded=2),
        "B_drop_248_only": dict(Normal=2, SQLi=1, CodeInj=1, CmdI=2, ambiguous=1, excluded=3),
    }
    assert sum(result["label_counts"].values()) == sum(int(k) * v for k, v in result["cardinality"].items())
    for label, count in result["label_counts"].items():
        assert result["cooccurrence"][label][label] == count
        for other in result["label_columns"]:
            assert result["cooccurrence"][label][other] == result["cooccurrence"][other][label]
    assert result["text_field_empty_rate"] == dict.fromkeys(TEXT_FIELDS, 0.5)
    pd.testing.assert_frame_equal(sample, original)


def test_empty_frame_has_undefined_rates(sample):
    result = profile_labels(sample.iloc[:0])
    assert result["cmd_injection_structure"]["p_248_given_88"] is None
    assert result["cmd_injection_structure"]["p_88_given_248"] is None
    assert all(value is None for value in result["text_field_empty_rate"].values())
    assert all(sum(counts.values()) == 0 for counts in result["mapping_scenarios"].values())


def test_missing_file_guidance(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(profile_srbh, "RAW_DIR", tmp_path)
    monkeypatch.setattr(profile_srbh.sys, "argv", ["profile_srbh.py"])
    assert profile_srbh.main() == 1
    assert "python src/data/download_srbh.py" in capsys.readouterr().err
