"""실데이터 학습 없이 표현·태그·CV 라우터 저장 계약을 확인한다."""
import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.eval import cascade_tradeoff, cross_validate, extract_embeddings, gate_g3_cv, gate_g2
from src.models.cnn import PayloadCNN


def test_tradeoff_tag_compatible_and_stage2_suffix(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tradeoff", "--balance"])
    args = cascade_tradeoff.parse_args()
    encoders = tuple(args.rgb_encoders.split(","))
    assert cascade_tradeoff.result_tag(args, encoders) == (
        "srbh_4class_cnn_raw_rgb_bal_len2304_nall_s42_dcuda-cpu_b1-32-128-1024")
    args.stage1_no_balance = True
    assert cascade_tradeoff.result_tag(args, encoders) == (
        "srbh_4class_cnn_raw_rgb_s2bal_len2304_nall_s42_dcuda-cpu_b1-32-128-1024")
    monkeypatch.setattr(sys, "argv", ["tradeoff"])
    args = cascade_tradeoff.parse_args()
    assert cascade_tradeoff.result_tag(args, encoders) == (
        "srbh_4class_cnn_raw_rgb_len2304_nall_s42_dcuda-cpu_b1-32-128-1024")


def test_stage1_flag_requires_balance(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tradeoff", "--stage1-no-balance"])
    with pytest.raises(SystemExit) as error:
        cascade_tradeoff.parse_args()
    assert error.value.code == 2


def test_gap_probabilities_match_eval_forward():
    torch.manual_seed(42)
    net = PayloadCNN(num_classes=4, in_channels=3).eval()
    x = torch.rand(5, 3, 48, 48)
    emb, probs = extract_embeddings.extract_loader_embeddings(
        net, DataLoader(TensorDataset(x), batch_size=2), "cpu")
    with torch.no_grad():
        expected = torch.softmax(net(x), 1).numpy()
    assert emb.shape == (5, 128) and emb.dtype == np.float32
    np.testing.assert_allclose(probs, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("model,extra", [("cnn", []), ("charcnn", ["--save-probs"])])
def test_cv_embeddings_invalid_cli(model, extra, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cv", "--model", model, "--save-embeddings", *extra])
    with pytest.raises(SystemExit) as error:
        cross_validate.main()
    assert error.value.code == 2


def test_cv_embeddings_tag():
    assert cross_validate.embeddings_tag("cnn_rgb_sealed", False) == "cnn_rgb_sealed"
    assert cross_validate.embeddings_tag("cnn_rgb_sealed", True) == "cnn_rgb_sealed_emb"


@pytest.mark.parametrize("problem", ["label", "prob", "class"])
def test_fixed_prob_alignment_rejected(problem):
    arrays = {"classes": np.array(["Normal", "Attack"]),
              "y_val": np.array([0, 1]), "y_test": np.array([0, 1])}
    probs = {split: np.array([[.8, .2], [.3, .7]]) for split in ("val", "test")}
    saved = {**arrays, **{f"p1_{split}": probs[split] for split in probs}}
    extract_embeddings.validate_saved_probs(saved, arrays, probs)
    if problem == "label":
        saved["y_val"] = np.array([1, 0])
    elif problem == "class":
        saved["classes"] = saved["classes"][::-1]
    else:
        saved["p1_val"] = saved["p1_val"][::-1]
    with pytest.raises(ValueError):
        extract_embeddings.validate_saved_probs(saved, arrays, probs)


def synthetic_cv(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 100)
    classes = ["Attack", "Normal"]
    arrays = {"classes": np.array(classes), "y": y,
              "label_fingerprint": np.array(cross_validate.label_fingerprint(y))}
    folds = []
    for k in range(1, 6):
        fold = {}
        for split, idx in (("test", np.arange((k-1)*40, k*40)),
                           ("val", np.arange((k % 5)*40, (k % 5+1)*40))):
            # 반분 학습 양쪽에 오답과 정답이 충분히 생기도록 확률을 구성한다.
            predicted = np.where(np.arange(40) % 3 == 0, 1-y[idx], y[idx])
            confidence = rng.uniform(.55, .95, 40)
            p1 = np.empty((40, 2))
            p1[np.arange(40), predicted] = confidence
            p1[np.arange(40), 1-predicted] = 1-confidence
            arrays[f"{split}_idx_{k}"] = idx
            arrays[f"{split}_probs_{k}"] = p1
            fold[f"{split}_idx"] = idx
            fold[f"{split}_emb"] = rng.normal(size=(40, 128)).astype(np.float32)
        folds.append(fold)
    first, second, embeddings = [tmp_path / f"{name}.npz" for name in ("first", "second", "embeddings")]
    np.savez(first, **arrays)
    for k in range(1, 6):
        for split in ("val", "test"):
            truth = y[arrays[f"{split}_idx_{k}"]]
            arrays[f"{split}_probs_{k}"] = np.eye(2)[truth] * .8 + .1
    np.savez(second, **arrays)
    cross_validate.save_fold_embeddings(embeddings, classes, y, folds)
    monkeypatch.setattr(cross_validate, "load_text_pool", lambda *a, **kw: (
        [f"id={i}%{i % 3}" for i in range(len(y))], y, classes))
    return first, second, embeddings


def test_g3_cv_cli_summary_and_overwrite(tmp_path, monkeypatch, capsys):
    first, second, emb = synthetic_cv(tmp_path, monkeypatch)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["g3cv", "--stage1", str(first), "--stage2", str(second),
                                      "--embeddings", str(emb), "--step", ".25", "--out", str(output)])
    assert gate_g3_cv.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == result["summary"]
    assert {"folds", "reduction", "p", "H-G3F-cv", "operating_points"} <= result["summary"].keys()
    assert len(result["folds"]) == 5
    assert result["summary"]["comparisons"]["g3_lite"]["exploratory"]
    for name in gate_g3_cv.GATES:
        assert "mean_r_N" in result["summary"]["operating_points"][name]
    before = output.read_bytes()
    assert gate_g3_cv.main() == 1
    assert output.read_bytes() == before


@pytest.mark.parametrize("problem", ["fingerprint", "index", "shape", "nan"])
def test_cv_embedding_alignment_rejected(tmp_path, monkeypatch, problem):
    first_path, second, path = synthetic_cv(tmp_path, monkeypatch)
    first, _, folds = gate_g3_cv.gate_cv.validate_alignment(first_path, second)
    with np.load(path) as saved:
        arrays = dict(saved)
    if problem == "fingerprint":
        arrays["label_fingerprint"] = np.array("wrong")
    elif problem == "index":
        arrays["val_idx_1"] = arrays["val_idx_1"][::-1]
    elif problem == "shape":
        arrays["test_emb_1"] = arrays["test_emb_1"][:, :127]
    else:
        arrays["test_emb_1"][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        gate_g3_cv.load_embeddings(path, first, folds)


def test_cv_fallback_applies_to_all_gates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    paths = synthetic_cv(tmp_path, monkeypatch)
    original = gate_g2.fold_curves
    def unreached(*args):
        rows = original(*args)
        for point in rows[0]["curves"]["b0"]:
            point["test_macro_f1"] = .997
        return rows
    monkeypatch.setattr(gate_g2, "fold_curves", unreached)
    result = gate_g3_cv.analyze(SimpleNamespace(stage1=paths[0], stage2=paths[1],
        embeddings=paths[2], track="srbh_4class", step=.25, seed=42))
    assert result["used_offset"] == .005 and 1 in result["initial_b0_unreached_folds"]
    assert all(row["target_macro_f1"] == .995 for row in result["folds"])


def test_embedding_empty_loader_rejected():
    net = PayloadCNN(num_classes=2, in_channels=3)
    with pytest.raises(ValueError, match="빈 입력"):
        extract_embeddings.extract_loader_embeddings(
            net, DataLoader(TensorDataset(torch.empty(0, 3, 48, 48))), "cpu")


def test_cv_fold_embeddings_follow_probability_order(monkeypatch):
    from types import SimpleNamespace
    import train
    # 학습을 대신해 이미 복원된 작은 모델을 반환하고 실제 평가 경로만 확인한다.
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "fit", lambda model, *a, **kw: (model.eval(), .5, 1))
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, (12, 48, 48, 3), dtype=np.uint8)
    labels = np.tile([0, 1], 6)
    tr, va, te = np.arange(4), np.array([7, 4, 6, 5]), np.array([11, 8, 10, 9])
    args = SimpleNamespace(batch_size=3, patch="8x8", epochs=1, lr=.001,
                           patience=1, save_probs=True, save_embeddings=True)
    saved = []
    cross_validate.run_fold_torch("cnn", images, labels, ["Attack", "Normal"],
        tr, va, te, args, is_image=True, fold_probabilities=saved)
    assert len(saved) == 1
    np.testing.assert_array_equal(saved[0]["val_idx"], va)
    np.testing.assert_array_equal(saved[0]["test_idx"], te)
    for split in ("val", "test"):
        assert saved[0][f"{split}_emb"].shape == (4, 128)
        assert saved[0][f"{split}_probs"].shape == (4, 2)


@pytest.mark.parametrize("prefix,ext", [("cv", "json"), ("cvprobs", "npz"), ("cvemb", "npz")])
def test_cv_embeddings_preflight_uses_new_names(tmp_path, monkeypatch, prefix, ext):
    monkeypatch.setattr(cross_validate, "RESULTS_DIR", tmp_path)
    existing = tmp_path / f"{prefix}_srbh_4class_cnn_raw_rgb_sealed_emb.{ext}"
    existing.write_bytes(b"preserved")
    monkeypatch.setattr(cross_validate, "load_image_pool", lambda *a, **kw: pytest.fail("충돌 후 풀 로딩 금지"))
    monkeypatch.setattr(sys, "argv", ["cv", "--model", "cnn", "--track", "srbh_4class",
        "--channels", "rgb", "--exclude-test", "--save-probs", "--save-embeddings"])
    with pytest.raises(SystemExit) as error:
        cross_validate.main()
    assert error.value.code == 2 and existing.read_bytes() == b"preserved"
