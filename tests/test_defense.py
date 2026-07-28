"""
Phase 11 (RQ3) — 방어 파이프라인 단위 테스트

여기서 지키려는 것 3가지(docs/10 §7):
  1. 계열 분할이 mutations.REGISTRY 와 어긋나지 않는다 → held-out 이 조용히 새지 않는다.
  2. 증강이 '치환'이라 클래스 비율·데이터 크기가 변하지 않는다 → 방어 효과와 분포 변화가 안 섞인다.
  3. ⚠️ npz 이미지와 학습 시점 온더플라이 이미지가 **바이트 단위로 같다**
     → 이게 깨지면 "원본은 npz, 증강본은 온더플라이"가 새 shortcut 이 되어 Phase 11 수치 전부 무효.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import augment as AUG
import mutations as MU
from tagging import build_tag, defense_suffix


# ---------------------------------------------------------------------------
# 1) 계열 분할
# ---------------------------------------------------------------------------
def test_families_partition_registry():
    """모든 변형이 정확히 한 계열에 속한다(누락·중복 없음)."""
    listed = [n for names in AUG.FAMILIES.values() for n in names]
    assert len(listed) == len(set(listed)), "변형이 두 계열에 중복 배치됨"
    assert set(listed) == set(MU.REGISTRY), "FAMILIES 와 mutations.REGISTRY 불일치"


def test_split_S0_is_everything_and_SA_holds_out_encoding():
    assert set(AUG.split_techniques("S0")) == set(MU.REGISTRY)
    assert AUG.held_out_techniques("S0") == []

    sa_seen = set(AUG.split_techniques("SA"))
    assert set(AUG.held_out_techniques("SA")) == set(AUG.FAMILIES["E"])
    # 인코딩 계열은 SA 학습에 단 하나도 들어가면 안 된다(가장 중요한 누수 지점).
    assert sa_seen.isdisjoint(AUG.FAMILIES["E"])


# ---------------------------------------------------------------------------
# 2) 치환식 증강
# ---------------------------------------------------------------------------
def _toy_corpus():
    """공격 3클래스 + Normal 로 이루어진 작은 코퍼스."""
    texts = ([f"' OR 1=1 -- {i}" for i in range(10)]
             + [f"<script>alert({i})</script>" for i in range(10)]
             + [f"; cat /etc/passwd {i}" for i in range(10)]
             + [f"id=55&nombre=jamon{i}" for i in range(10)])
    labels = (["SQLInjection"] * 10 + ["XSS"] * 10
              + ["CommandInjection"] * 10 + ["Normal"] * 10)
    return texts, labels


def test_augment_replaces_exact_ratio_and_keeps_size():
    texts, labels = _toy_corpus()
    out, info = AUG.augment_texts(texts, labels, mutation_split="S0", ratio=0.5,
                                  budget=3, seed=42)
    assert len(out) == len(texts)                    # 크기 불변 = 치환
    assert info["n_replaced"] == 15                  # 공격 3클래스 × 10개 × 0.5
    assert info["per_label"] == {"SQLInjection": 5, "XSS": 5, "CommandInjection": 5}


def test_augment_never_touches_normal():
    """Normal 은 의미보존 변형이 정의되지 않았으므로 손대지 않는다(클래스 비율 보존의 근거)."""
    texts, labels = _toy_corpus()
    out, _ = AUG.augment_texts(texts, labels, mutation_split="S0", ratio=1.0,
                               budget=3, seed=7)
    for original, mutated, label in zip(texts, out, labels):
        if label == "Normal":
            assert original == mutated


def test_augment_ratio_zero_is_noop():
    texts, labels = _toy_corpus()
    out, info = AUG.augment_texts(texts, labels, ratio=0.0, seed=1)
    assert out == texts and info["n_replaced"] == 0


def test_augment_is_deterministic():
    texts, labels = _toy_corpus()
    a, _ = AUG.augment_texts(texts, labels, mutation_split="SA", ratio=0.5, seed=42)
    b, _ = AUG.augment_texts(texts, labels, mutation_split="SA", ratio=0.5, seed=42)
    c, _ = AUG.augment_texts(texts, labels, mutation_split="SA", ratio=0.5, seed=43)
    assert a == b, "같은 시드인데 결과가 다르다(재현성 붕괴)"
    assert a != c, "시드를 바꿔도 결과가 같다(시드가 실제로 안 쓰이는 중)"


def test_heldout_encoding_never_appears_in_SA_augmentation():
    """SA 로 증강한 텍스트에는 인코딩 계열의 흔적(%XX, &#NN)이 새로 생기지 않아야 한다.

    기법 이름 목록만 확인하면 '레지스트리는 맞는데 실제로는 인코딩이 적용되는' 버그를
    못 잡는다. 그래서 산출 문자열 자체를 본다(원본에 없던 표식이 생겼는지).
    """
    texts, labels = _toy_corpus()
    out, _ = AUG.augment_texts(texts, labels, mutation_split="SA", ratio=1.0,
                               budget=3, seed=42)
    for original, mutated in zip(texts, out):
        if "%" not in original:
            assert "%" not in mutated, f"held-out URL 인코딩이 샜다: {mutated!r}"
        if "&#" not in original:
            assert "&#" not in mutated, f"held-out 엔티티 인코딩이 샜다: {mutated!r}"


def test_augment_rejects_bad_arguments():
    texts, labels = _toy_corpus()
    for bad in (-0.1, 1.5):
        try:
            AUG.augment_texts(texts, labels, ratio=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ratio={bad} 인데 예외가 없다")
    try:
        AUG.augment_texts(texts, labels, mutation_split="없는분할")
    except ValueError:
        pass
    else:
        raise AssertionError("알 수 없는 분할인데 예외가 없다")


# ---------------------------------------------------------------------------
# 3) ⚠️ npz ↔ 온더플라이 이미지 동일성 (Phase 11 의 생명선)
# ---------------------------------------------------------------------------
def _write_toy_csv(dir_path, track="toy", split="train"):
    """길이·문자 구성이 제각각인 페이로드(짧음/긺/비ASCII/빈 문자열)로 CSV 를 만든다."""
    texts = ["' OR 1=1 --", "<script>alert(1)</script>", "", "A" * 5000, "표류 ünïcode"]
    df = pd.DataFrame({"text_raw": texts,
                       "text_decoded": texts,
                       "label": ["SQLInjection", "XSS", "Normal", "SQLInjection", "Normal"]})
    dir_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(dir_path / f"{track}_{split}.csv", index=False, encoding="utf-8")
    return texts


def _build_and_compare(tmp_path, monkeypatch, channels, encoders):
    import build_image_dataset as BID
    import train as T

    processed, images = tmp_path / "processed", tmp_path / "images"
    texts = _write_toy_csv(processed)
    monkeypatch.setattr(BID, "PROCESSED_DIR", processed)
    monkeypatch.setattr(BID, "IMAGES_DIR", images)

    out_path, n = BID.build_split("toy", "train", "text_raw", 48, channels, encoders)
    assert n == len(texts)
    from_npz = np.load(out_path, allow_pickle=False)["images"]
    on_the_fly = T.texts_to_images(texts, 48, channels, encoders)

    assert from_npz.shape == on_the_fly.shape
    assert from_npz.dtype == on_the_fly.dtype
    assert np.array_equal(from_npz, on_the_fly), (
        f"{channels}: npz 와 온더플라이 이미지가 다르다 — "
        "증강 학습이 두 경로를 섞으면 '증강본 여부'가 새 shortcut 이 된다(docs/10 §7.1)")


def test_gray_npz_matches_on_the_fly(tmp_path, monkeypatch):
    _build_and_compare(tmp_path, monkeypatch, "gray", ("raw_byte", "char_class", "local_entropy"))


def test_rgb_npz_matches_on_the_fly(tmp_path, monkeypatch):
    _build_and_compare(tmp_path, monkeypatch, "rgb", ("raw_byte", "char_class", "local_entropy"))


def test_rgb_custom_encoders_npz_matches_on_the_fly(tmp_path, monkeypatch):
    """기본 조합이 아닌 ablation 조합에서도 두 경로가 일치해야 한다."""
    _build_and_compare(tmp_path, monkeypatch, "rgb", ("raw_byte", "char_class", "byte_delta"))


# ---------------------------------------------------------------------------
# 4) tag 규칙 — 기존 산출물과의 호환이 깨지면 과거 결과를 잃는다
# ---------------------------------------------------------------------------
def test_tag_backward_compatible_for_existing_experiments():
    """Phase 4~10 에서 쓰던 조합은 방어 축 도입 후에도 **글자 그대로** 같아야 한다."""
    assert build_tag("payload_4class", "cnn", "raw") == "payload_4class_cnn_raw"
    assert (build_tag("payload_4class_csicnorm", "charcnn", "raw", balance=True)
            == "payload_4class_csicnorm_charcnn_raw_bal")
    assert (build_tag("payload_4class", "cnn", "raw", channels="rgb",
                      encoders=("raw_byte", "char_class", "byte_delta"))
            == "payload_4class_cnn_raw_rgb-rb-cc-bd")
    assert (build_tag("payload_4class", "vit", "raw", patch="1x48", lr=3e-4)
            == "payload_4class_vit_raw_p1x48_lr0.0003")
    # 기본 lr 은 접미사가 붙지 않는다(과거 파일명 호환의 핵심 규칙)
    assert build_tag("payload_4class", "cnn", "raw", lr=1e-3) == "payload_4class_cnn_raw"


def test_defense_suffix_axes():
    assert defense_suffix("none") == ""
    assert defense_suffix("norm") == "_def-norm"
    assert defense_suffix("advtrain", "SA", 0.5) == "_def-advtrain_SA_r0.5"
    # 분할/비율이 다르면 반드시 다른 파일이어야 한다(상호 덮어쓰기 방지)
    assert defense_suffix("advtrain", "S0", 0.25) != defense_suffix("advtrain", "SA", 0.25)
    assert defense_suffix("advtrain", "SA", 0.25) != defense_suffix("advtrain", "SA", 0.5)


def test_defense_tag_full_composition():
    assert (build_tag("payload_4class_csicnorm", "cnn", "raw", channels="rgb",
                      balance=True, defense="advtrain", mutation_split="SA", aug_ratio=0.5)
            == "payload_4class_csicnorm_cnn_raw_rgb_def-advtrain_SA_r0.5_bal")
    assert (build_tag("payload_4class_csicnorm", "charcnn", "decoded",
                      balance=True, defense="norm")
            == "payload_4class_csicnorm_charcnn_decoded_def-norm_bal")
