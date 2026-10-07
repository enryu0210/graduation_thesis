"""T3 판정: 같은 지연 이하의 char-CNN 점보다 F1 이 높을 때만 대안으로 병기한다."""
import numpy as np

from src.eval.stage2_candidates import budget_masks, candidate_curve, dominance


def test_dominance_lists_only_cheaper_and_better_candidate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p1 = np.eye(2)[1 - y] * 0.4 + 0.3            # 1차는 전부 틀린다(확신도 0.7)
    p1[:, :] += rng.uniform(0, 0.01, p1.shape)    # 넘김 순서를 정하는 동률 깨기
    perfect, noisy = np.eye(2)[y], np.eye(2)[np.where(rng.random(400) < 0.5, y, 1 - y)]
    masks = budget_masks(p1, p1, 0.25)
    curve = lambda p2, t2: candidate_curve(masks, p1, p1, p2, p2, y, y, 0, 0.1, t2)
    curves = {"charcnn": curve(noisy, 1.0), "bilstm": curve(noisy, 1.0), "tfidf": curve(perfect, 1.0)}
    preds2 = {k: p.argmax(1) for k, p in (("charcnn", noisy), ("bilstm", noisy), ("tfidf", perfect))}
    result = dominance(curves, masks, preds2, p1, y, 2, 200, 0)
    assert result["tfidf"]["listed_as_alternative"]      # 같은 지연에서 더 정확하다
    assert result["bilstm"]["listed_as_alternative"] is False   # char-CNN 과 같으면 이득 0 → 병기하지 않는다
