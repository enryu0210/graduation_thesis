"""합성 확률로 저장 계약과 선택적 추론을 학습 없이 검증한다."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from src.eval import cross_validate as cv


def test_save_fold_probabilities_format_and_order(tmp_path):
    classes = ["Normal", "SQLInjection"]
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    folds = [
        {"val_idx": [3, 0], "val_probs": [[.2, .8], [.9, .1]],
         "test_idx": [1], "test_probs": [[.3, .7]]},
        {"val_idx": [1], "val_probs": [[.4, .6]],
         "test_idx": [2, 0], "test_probs": [[.8, .2], [.7, .3]]},
    ]
    path = tmp_path / "results" / "cvprobs_synthetic.npz"
    cv.save_fold_probabilities(path, classes, y, folds)
    with np.load(path, allow_pickle=False) as saved:
        expected_keys = {"classes", "label_fingerprint", "y"}
        expected_keys.update(f"{name}_{k}" for k in (1, 2)
                             for name in ("val_idx", "val_probs", "test_idx", "test_probs"))
        assert set(saved.files) == expected_keys
        assert saved["classes"].dtype.kind == "U"
        assert saved["classes"].tolist() == classes
        assert saved["label_fingerprint"].shape == ()
        assert saved["label_fingerprint"].dtype.kind == "U"
        assert saved["label_fingerprint"].item() == cv.label_fingerprint(y)
        assert saved["y"].dtype == np.int64
        np.testing.assert_array_equal(saved["y"], y)
        for k, fold in enumerate(folds, start=1):
            for split in ("val", "test"):
                indices = saved[f"{split}_idx_{k}"]
                probabilities = saved[f"{split}_probs_{k}"]
                assert indices.dtype == np.int64
                assert probabilities.dtype == np.float32
                assert probabilities.shape == (len(indices), len(classes))
                np.testing.assert_array_equal(indices, fold[f"{split}_idx"])
                np.testing.assert_array_equal(probabilities,
                                              np.asarray(fold[f"{split}_probs"], dtype=np.float32))


@pytest.mark.parametrize("indices,probabilities", [
    ([0, 1], [[.5, .5]]), ([0], [[1.0]]), ([-1], [[.5, .5]]),
    ([2], [[.5, .5]]), ([[0]], [[.5, .5]]),
])
def test_save_fold_probabilities_rejects_invalid_alignment(tmp_path, indices, probabilities):
    fold = {"val_idx": indices, "val_probs": probabilities,
            "test_idx": [1], "test_probs": [[.5, .5]]}
    with pytest.raises(ValueError):
        cv.save_fold_probabilities(tmp_path / "invalid.npz", ["A", "B"], [0, 1], [fold])
    assert not (tmp_path / "invalid.npz").exists()


@pytest.mark.parametrize("model", sorted(cv.TFIDF_MODELS))
def test_save_probs_rejects_tfidf_before_loading_data(monkeypatch, model):
    monkeypatch.setattr(sys, "argv", ["cross_validate", "--model", model, "--save-probs"])
    with pytest.raises(SystemExit) as error:
        cv.main()
    assert error.value.code == 2


@pytest.mark.parametrize("save_probs", [False, True])
def test_torch_fold_only_evaluates_val_when_requested(monkeypatch, save_probs):
    import torch
    import train as T

    # 학습과 지표는 대체하고 실제 DataLoader 순서만 확인해 인덱스 대응을 검사한다.
    monkeypatch.setattr(T, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(T, "build_model", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(T, "fit", lambda model, *args, **kwargs: (model, .8, 1))
    monkeypatch.setattr(cv.M, "compute_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen_labels = []
    scores = []

    def evaluate(model, loader, device):
        labels = np.concatenate([batch_y.numpy() for _, batch_y in loader])
        seen_labels.append(labels.tolist())
        probabilities = np.array([[.8, .2] if label == 0 else [.3, .7]
                                  for label in labels], dtype=np.float32)
        scores.append(probabilities)
        return labels, labels, probabilities

    monkeypatch.setattr(T, "evaluate", evaluate)
    args = SimpleNamespace(batch_size=2, patch="8x8", epochs=1, lr=.001,
                           patience=1, save_probs=save_probs)
    collector = [] if save_probs else None
    result = cv.run_fold_torch("charcnn", np.zeros((6, 4), dtype=np.int16),
                               np.array([0, 1, 0, 1, 0, 1]), ["A", "B"],
                               np.array([0, 1]), np.array([5, 2]), np.array([4, 3]),
                               args, is_image=False, fold_probabilities=collector)
    assert seen_labels == ([[0, 1], [1, 0]] if save_probs else [[0, 1]])
    assert set(result) == {"throughput_samples_per_sec", "best_val_macro_f1", "best_epoch"}
    if save_probs:
        assert len(collector) == 1
        np.testing.assert_array_equal(collector[0]["val_idx"], [5, 2])
        np.testing.assert_array_equal(collector[0]["test_idx"], [4, 3])
        assert collector[0]["test_probs"] is scores[0]
        assert collector[0]["val_probs"] is scores[1]
