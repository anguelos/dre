import random
import sys

import numpy as np
import pytest
import torch

from phoc import PHOCToSemantic
from phoc import train_semantic


def test_model_outputs_unit_vectors():
    model = PHOCToSemantic(output_dim=8, hidden=(16,), levels=(1, 2), max_len=5)
    out = model(["register", "regster", "x"])
    assert out.shape == (3, 8)
    assert torch.allclose(out.norm(dim=1), torch.ones(3))
    assert model.config()["hidden"] == [16]


def test_custom_mapping_is_kept_in_config():
    model = PHOCToSemantic(output_dim=4, hidden=(), mapping={c: c for c in "abc"})
    assert model.phoc.n_symbols == 3 and model.config()["mapping"] == {"a": "a", "b": "b", "c": "c"}


def test_corrupt_is_deterministic_and_bounded():
    assert train_semantic.corrupt("register", 0.0, random.Random(0)) == "register"
    noisy = [train_semantic.corrupt("register", 0.5, random.Random(seed)) for seed in range(20)]
    assert noisy == [train_semantic.corrupt("register", 0.5, random.Random(seed)) for seed in range(20)]
    assert any(n != "register" for n in noisy) and all(n for n in noisy)
    assert train_semantic.corrupt("a", 1.0, random.Random(3))


@pytest.fixture
def vectors_file(tmp_path):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "x1", "a"]
    lines = [w + " " + " ".join(f"{v:.4f}" for v in rng.normal(size=6)) for w in words]
    lines.append("zero " + " ".join(["0"] * 6))
    path = tmp_path / "vectors.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def run(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["train_semantic", *arguments])
    train_semantic.main()


def test_train_resume_and_reload(tmp_path, vectors_file, monkeypatch, capsys):
    output = tmp_path / "model.pt"
    common = [f"--vectors={vectors_file}", f"--output={output}", "--hidden=16", "--batch_size=4", "--device=cpu", "--val_fraction=0.2"]
    run(monkeypatch, *common, "--epochs=2")
    state = torch.load(output, weights_only=False)
    assert state["epoch"] == 2 and len(state["history"]) == 2
    assert set(state["split"]["train"]) | set(state["split"]["val"]) == {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda"}
    run(monkeypatch, *common, "--epochs=3")
    assert "resuming" in capsys.readouterr().out
    assert torch.load(output, weights_only=False)["epoch"] == 3
    with pytest.raises(SystemExit):
        run(monkeypatch, *["--hidden=8" if a.startswith("--hidden") else a for a in common], "--epochs=4")
    model = PHOCToSemantic.from_checkpoint(str(output), weights="last")
    assert model(["alpha"]).shape == (1, 6)
    assert not model.training


def test_training_losses():
    targets = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
    assert train_semantic.training_loss(targets, targets, "cosine", 0.05) < 1e-6
    assert train_semantic.training_loss(targets.roll(1, 0), targets, "cosine", 0.05) > 0
    right = train_semantic.training_loss(targets, targets, "contrastive", 0.05)
    wrong = train_semantic.training_loss(targets.roll(1, 0), targets, "contrastive", 0.05)
    assert right < wrong
    with pytest.raises(SystemExit):
        train_semantic.training_loss(targets, targets, "mse", 0.05)


def test_contrastive_training_and_old_state_files(tmp_path, vectors_file, monkeypatch):
    output = tmp_path / "c.pt"
    common = [f"--vectors={vectors_file}", f"--output={output}", "--hidden=16", "--batch_size=4", "--device=cpu", "--loss=contrastive"]
    run(monkeypatch, *common, "--epochs=1")
    assert torch.load(output, weights_only=False)["config"]["loss"] == "contrastive"
    with pytest.raises(SystemExit):
        run(monkeypatch, *common[:-1], "--loss=cosine", "--epochs=2")
    with pytest.raises(SystemExit):
        run(monkeypatch, *common[:-1], "--loss=mse", "--epochs=2")
    # A state file written before the loss settings existed resumes as the default (cosine) loss.
    state = torch.load(output, weights_only=False)
    del state["config"]["loss"], state["config"]["contrastive_temperature"]
    torch.save(state, output)
    run(monkeypatch, *common[:-1], "--epochs=2")
    assert torch.load(output, weights_only=False)["epoch"] == 2


def test_npy_vectors_and_missing_arguments(tmp_path, monkeypatch):
    np.save(tmp_path / "v.npy", np.random.default_rng(1).normal(size=(6, 4)).astype(np.float32))
    (tmp_path / "v.txt").write_text("one\ntwo\nthree\nfour\nfive\nsix\n")
    output = tmp_path / "m.pt"
    run(monkeypatch, f"--vectors={tmp_path / 'v.npy'}", f"--vocabulary={tmp_path / 'v.txt'}", f"--output={output}", "--hidden=8", "--epochs=1",
        "--device=cpu", "--max_words=5", "-vv")
    assert len(torch.load(output, weights_only=False)["split"]["train"]) == 4
    with pytest.raises(SystemExit):
        run(monkeypatch, f"--vectors={tmp_path / 'v.npy'}", f"--output={output}")
    torch.save({"format": "other"}, tmp_path / "other.pt")
    with pytest.raises(SystemExit):
        run(monkeypatch, f"--vectors={tmp_path / 'v.npy'}", f"--vocabulary={tmp_path / 'v.txt'}", f"--output={tmp_path / 'other.pt'}", "--device=cpu")
