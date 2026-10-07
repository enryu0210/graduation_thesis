"""G5 변형 매핑: 허용 클래스에만 적용하고 원문이 안 바뀐 행은 버린다."""
from src.eval.gate_g5_mutation import LABEL_MUTATIONS, mutate_rows


def test_mapping_and_unchanged_rows_dropped():
    assert "url_encode" in LABEL_MUTATIONS["CodeInjection"]
    assert not set(LABEL_MUTATIONS["CodeInjection"]) - set(LABEL_MUTATIONS["SQLInjection"])
    texts = ["a b", "a b", "ab", "a b"]
    labels = ["SQLInjection", "Normal", "CodeInjection", "CodeInjection"]
    rows, mutated = mutate_rows(texts, labels, "space_to_tab", seed=42)
    # Normal 은 대상 밖, "ab" 는 공백이 없어 그대로라 빠진다.
    assert rows.tolist() == [0, 3] and all("\t" in m for m in mutated)
    rows, _ = mutate_rows(texts, labels, next(m for m in LABEL_MUTATIONS["SQLInjection"]
                                              if m not in LABEL_MUTATIONS["CodeInjection"]), seed=42)
    assert set(rows.tolist()) <= {0}   # SQLi 전용 변형은 CodeInjection 행에 안 걸린다
