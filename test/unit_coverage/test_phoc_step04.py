import sys

import numpy as np
import pytest
import torch

from phoc import PHOCToString
from phoc import train_decoder


def test_targets_and_shapes():
    model = PHOCToString(alphabet="abc", hidden=16, embedding=4, condition=8, levels=(1, 2), max_len=5, max_output=6)
    targets = model.encode_targets(["ab", "cab", "axb"])
    assert targets.tolist() == [[2, 3, 0, 0], [4, 2, 3, 0], [2, 3, 0, 0]]  # "x" is not in the alphabet; 0 ends a word
    assert model(["ab", "cab", "axb"], targets).shape == (3, 4, 5)
    spelled = model.spell(["ab", "cab"])
    assert len(spelled) == 2 and all(len(s) <= 6 and set(s) <= set("abc") for s in spelled)
    assert model.config()["alphabet"] == "abc"


def test_decoder_learns_to_copy():
    torch.manual_seed(0)
    words = ["ab", "ba", "abc", "cab", "bca", "cc"]
    model = PHOCToString(alphabet="abc", hidden=32, embedding=8, condition=16, levels=(1, 2, 3), max_len=4, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    targets = model.encode_targets(words)
    for _ in range(200):
        logits = model(words, targets)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert model.eval().spell(words) == words


def test_spelling_metrics():
    assert train_decoder.edit_distance("register", "regster") == 1
    assert train_decoder.edit_distance("", "abc") == 3
    metrics = train_decoder.spelling_metrics(["abc", "abd"], ["abc", "abc"])
    assert metrics == {"accuracy": 0.5, "cer": 1 / 6}


@pytest.fixture
def vectors_file(tmp_path):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda"]
    path = tmp_path / "vectors.txt"
    path.write_text("\n".join(w + " " + " ".join(f"{v:.3f}" for v in rng.normal(size=4)) for w in words) + "\n")
    return path


def run(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["train_decoder", *arguments])
    train_decoder.main()


def test_train_resume_and_reload(tmp_path, vectors_file, monkeypatch, capsys):
    output = tmp_path / "decoder.pt"
    common = [f"--vectors={vectors_file}", f"--output={output}", "--hidden=16", "--embedding=4", "--condition=8", "--batch_size=4", "--device=cpu", "-vv"]
    run(monkeypatch, *common, "--epochs=2")
    state = torch.load(output, weights_only=False)
    assert state["format"] == train_decoder.FORMAT and state["epoch"] == 2 and "unseen_noisy_cer" in state["history"][-1]
    run(monkeypatch, *common, "--epochs=3")
    assert "resuming" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run(monkeypatch, *["--hidden=8" if a.startswith("--hidden") else a for a in common], "--epochs=4")
    model = PHOCToString.from_checkpoint(str(output), weights="last")
    assert isinstance(model.spell(["alpha"])[0], str) and not model.training
    with pytest.raises(SystemExit):
        run(monkeypatch, f"--output={output}")
    torch.save({"format": "other"}, tmp_path / "other.pt")
    with pytest.raises(SystemExit):
        run(monkeypatch, f"--vectors={vectors_file}", f"--output={tmp_path / 'other.pt'}", "--device=cpu")


def test_input_augmentation_only_while_training():
    torch.manual_seed(0)
    model = PHOCToString(alphabet="abc", hidden=16, embedding=4, condition=8, levels=(1, 2), max_len=5, input_dropout=0.5, input_noise=1.0)
    assert model.config()["input_dropout"] == 0.5 and model.config()["input_noise"] == 1.0
    model.eval()
    assert torch.equal(model._start(["abc"])[0], model._start(["abc"])[0])
    model.train()
    assert not torch.equal(model._start(["abc"])[0], model._start(["abc"])[0])


def test_old_state_files_resume_without_augmentation_settings(tmp_path, vectors_file, monkeypatch):
    output = tmp_path / "decoder.pt"
    common = [f"--vectors={vectors_file}", f"--output={output}", "--hidden=16", "--embedding=4", "--condition=8", "--batch_size=4", "--device=cpu"]
    run(monkeypatch, *common, "--epochs=1", "--input_noise=0.3")
    with pytest.raises(SystemExit):
        run(monkeypatch, *common, "--epochs=2")
    state = torch.load(output, weights_only=False)
    for key in ("input_dropout", "input_noise"):
        del state["config"][key], state["model_config"][key]
    torch.save(state, output)
    run(monkeypatch, *common, "--epochs=2")
    assert torch.load(output, weights_only=False)["epoch"] == 2
