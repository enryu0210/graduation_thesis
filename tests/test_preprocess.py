"""전처리·분할(preprocess.py)의 무결성 단위 테스트.

검증 포인트:
    - normalize_text 가 URL 디코딩(이중 인코딩 포함)과 HTML 엔티티 디코딩을 올바로 하는가
    - clean_and_split 이
        * 빈 문자열·완전 중복을 제거하는가
        * train/val/test 사이에 동일 페이로드가 새지 않는가(leakage 방지)
        * 대략 70/15/15 비율과 클래스 분포(stratify)를 유지하는가
"""

import pandas as pd

from data.preprocess import clean_and_split, normalize_text


def test_normalize_url_decoding():
    # %27 → ' , + → 공백
    assert normalize_text("a%27+OR+1%3D1") == "a' OR 1=1"


def test_normalize_double_encoding():
    # 이중 인코딩(%2527 → %27 → ')까지 반복 디코딩되어야 한다.
    assert normalize_text("%2527") == "'"


def test_normalize_html_entities():
    assert normalize_text("&lt;script&gt;") == "<script>"


def _make_synthetic_df(per_class: int = 200) -> pd.DataFrame:
    """클래스 2개, 각 per_class 개의 고유 페이로드 + 중복/빈값을 섞은 합성 데이터."""
    rows = []
    for cls in ("Normal", "Attack"):
        for i in range(per_class):
            rows.append({"text_raw": f"{cls}_payload_{i}", "label": cls})
    # 완전 중복 2개 + 빈 문자열 1개 추가 → 전처리에서 걸러져야 한다.
    rows.append({"text_raw": "Normal_payload_0", "label": "Normal"})
    rows.append({"text_raw": "Attack_payload_0", "label": "Attack"})
    rows.append({"text_raw": "   ", "label": "Normal"})
    return pd.DataFrame(rows)


def test_split_removes_duplicates_and_empty():
    df = _make_synthetic_df(per_class=200)
    out, _ = clean_and_split(df)
    # 원래 고유 400개 + 중복2 + 빈1 = 403 → 정제 후 정확히 400개여야 한다.
    assert len(out) == 400
    assert out["text_raw"].duplicated().sum() == 0


def test_split_no_leakage_between_splits():
    df = _make_synthetic_df(per_class=200)
    out, _ = clean_and_split(df)
    sets = {s: set(out[out["split"] == s]["text_raw"]) for s in ("train", "val", "test")}
    # 어떤 두 split 도 겹치는 페이로드가 없어야 한다.
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_split_ratios_and_stratify():
    df = _make_synthetic_df(per_class=200)
    out, _ = clean_and_split(df)
    counts = out["split"].value_counts()
    # 400개 기준 대략 280/60/60 (오차 소폭 허용)
    assert abs(counts["train"] - 280) <= 2
    assert abs(counts["val"] - 60) <= 2
    assert abs(counts["test"] - 60) <= 2
    # stratify: 각 split 에서 두 클래스가 거의 반반이어야 한다.
    for s in ("train", "val", "test"):
        part = out[out["split"] == s]
        normal = (part["label"] == "Normal").sum()
        attack = (part["label"] == "Attack").sum()
        assert abs(normal - attack) <= 2
