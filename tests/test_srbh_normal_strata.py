"""합성 UA로 세 층과 감사 제외의 순서를 고정한다."""

import pandas as pd
import pytest

from src.analysis.srbh_normal_strata import assign_normal_strata, fpr_by_stratum


def synthetic():
    codes = ["000", "66", "242", "88", "248", "126", "16", "34", "49", "100", "153", "272", "310", "549"]
    active = ["000"] * 5 + ["66", "88", "248"]
    raw = pd.DataFrame({f"{code} - 합성": [str(int(value == code)) for value in active] for code in codes})
    raw["request_http_request"] = ["a", "b", "c", "c", "a", "attack", "other", "last"]
    raw["request_body"] = ""
    raw["request_cookie"] = ""
    raw["request_user_agent"] = ["user", "scan", "user", "scan", "scan", "scan", "scan", "other"]
    track = {"test": pd.DataFrame({"row_id": [0, 1, 2, 5], "text_raw": ["a\n", "b\n", "c\n", "attack\n"],
                                   "label": ["Normal"] * 3 + ["SQLInjection"]})}
    return raw, pd.DataFrame({"row_id": [4]}), track


def test_three_strata_exclude_audit():
    frame, stats = assign_normal_strata(*synthetic())
    assert frame.normal_source.tolist() == ["real_user", "scanner", "mixed"]
    assert stats["scanner_ua"] == "scan"
    assert stats["by_split"]["test"] == {"real_user": 1, "scanner": 1, "mixed": 1}


def test_missing_original_raises():
    raw, flags, track = synthetic()
    track["test"].loc[0, "text_raw"] = "missing"
    with pytest.raises(ValueError, match="원본"):
        assign_normal_strata(raw, flags, track)


def test_scanner_population_precedes_conflict_removal():
    raw, flags, track = synthetic()
    # 공격 두 행이 Normal과 충돌해도 UA 최빈값 모집단에는 남아야 한다.
    raw.loc[[5, 6], "request_http_request"] = "a"
    _, stats = assign_normal_strata(raw, flags, track)
    assert stats["scanner_ua"] == "scan"


def test_fpr_totals_and_non_normal_exclusion():
    report = fpr_by_stratum(["Normal"] * 3 + ["SQLInjection"],
                            ["Normal", "SQLInjection", "CodeInjection", "Normal"],
                            ["real_user", "scanner", "mixed", ""])
    for key in ("n", "false_alarms"):
        assert sum(report[name][key] for name in ("real_user", "scanner", "mixed")) == report["all"][key]
    assert report["all"]["fpr"]["point"] == 2 / 3
    with pytest.raises(ValueError):
        fpr_by_stratum(["Normal"], [], [])
