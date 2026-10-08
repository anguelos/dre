# Train PHOCToString: spell a word from its PHOC, as a denoising task — input spellings are corrupted with OCR-like
# edits, targets are the clean spellings. The words and their training / held-out split are exactly those of
# train_semantic.py for the same settings, so the two models can be compared and chained. Validation reports word
# accuracy and character error rate (CER) on the same evaluation sets: unseen words clean and corrupted, and fresh
# corruptions of known words.
#
# As in train_semantic.py, one "fat" .pt file holds everything needed to resume or reuse, rewritten atomically after
# every epoch; running again with the same --output resumes until --epochs is reached.
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import fargv
import torch
import tqdm

from .step04_decoder import PHOCToString
from .train_semantic import corrupt, evaluation_sets, file_digest, load_vectors, make_split, save_atomically

FORMAT = "phoc_decoder/1"
# Settings that define the model or the data; a resumed run must not change them (the vectors are checked by content).
FIXED_SETTINGS = ("hidden", "embedding", "condition", "levels", "max_len", "dropout", "input_dropout", "input_noise", "val_fraction", "max_words",
                  "min_length", "seed")


@dataclass
class Config:
    """Train a PHOC → spelling decoder, resuming from --output if it exists."""
    vectors: str = ""
    "Word list source, as for train_semantic.py: GloVe-format text, or a .npy matrix with --vocabulary (only the words are used)."
    vocabulary: str = ""
    output: str = "phoc_decoder.pt"
    "Fat state file; resumed if it exists."
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    hidden: int = 512
    embedding: int = 64
    condition: int = 128
    "Size of the PHOC projection fed to the GRU at every step."
    dropout: float = 0.1
    input_dropout: float = 0.0
    "Training-time dropout on the PHOC input: probability of zeroing each cell."
    input_noise: float = 0.0
    "Training-time Gaussian noise on the PHOC input: standard deviation added to each cell."
    levels: str = "1,2,3,5"
    max_len: int = 20
    noise: float = 0.1
    "Probability that each character of an input spelling is substituted, deleted or followed by an inserted one."
    val_fraction: float = 0.1
    max_words: int = 0
    min_length: int = 2
    eval_size: int = 2000
    "Words per evaluation set checked after each epoch."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    verbosity: int = fargv.FargvInt(1, short_name="v", is_count_switch=True)
    "-v: per-epoch summary  -vv: progress bars."


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def spelling_metrics(predicted: List[str], truth: List[str]) -> Dict[str, float]:
    return {"accuracy": sum(p == t for p, t in zip(predicted, truth)) / len(truth),
            "cer": sum(edit_distance(p, t) for p, t in zip(predicted, truth)) / sum(len(t) for t in truth)}


def main() -> None:
    cfg, _ = fargv.parse(Config)
    if not cfg.vectors or (cfg.vectors.endswith(".npy") and not cfg.vocabulary):
        raise SystemExit("--vectors is required (with --vocabulary for a .npy matrix)")
    output = Path(cfg.output)
    vectors_digest = file_digest(cfg.vectors, cfg.vocabulary)
    words, _ = load_vectors(cfg)
    device = torch.device(cfg.device)

    if output.exists():
        state = torch.load(output, map_location="cpu", weights_only=False)  # our own file: it holds RNG states and history
        if state["format"] != FORMAT:
            raise SystemExit(f"{output} is not a {FORMAT} state file")
        defaults = {k: f.default for k, f in Config.__dataclass_fields__.items()}  # settings added after the file was made
        changed = [k for k in FIXED_SETTINGS if state["config"].get(k, defaults[k]) != asdict(cfg)[k]]
        if changed or state["vectors_sha256"] != vectors_digest:
            raise SystemExit(f"{output} was trained with different {changed or ['vectors']}; use another --output to start afresh")
        if cfg.verbosity >= 1:
            print(f"resuming {output} after epoch {state['epoch']}")
    else:
        split = make_split(words, cfg.val_fraction, cfg.seed)
        alphabet = "".join(sorted({c for w in split["train"] for c in w}))
        state = {"format": FORMAT, "epoch": 0, "history": [], "best_loss": float("inf"), "best_epoch": 0, "vectors_sha256": vectors_digest,
                 "split": split, "rng": None,
                 "model_config": {"alphabet": alphabet, "hidden": cfg.hidden, "embedding": cfg.embedding, "condition": cfg.condition,
                                  "levels": [int(level) for level in cfg.levels.split(",")], "max_len": cfg.max_len, "dropout": cfg.dropout,
                                  "input_dropout": cfg.input_dropout, "input_noise": cfg.input_noise, "max_output": max(len(w) for w in words) + 4}}
        torch.manual_seed(cfg.seed)
    state["config"] = asdict(cfg)

    model = PHOCToString(**state["model_config"]).to(device)
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if not name.startswith("phoc.")], lr=cfg.lr)
    if "model" in state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    rng = random.Random(cfg.seed)
    if state["rng"] is not None:
        rng.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])

    checks = {name: (inputs[:cfg.eval_size], truth[:cfg.eval_size]) for name, (inputs, truth) in evaluation_sets(state["split"], cfg.noise, cfg.seed).items()}
    train_words = state["split"]["train"]
    for epoch in range(state["epoch"] + 1, cfg.epochs + 1):
        started = time.time()
        model.train()
        order = list(train_words)
        rng.shuffle(order)
        batches = range(0, len(order), cfg.batch_size)
        batches = tqdm.tqdm(batches, desc=f"epoch {epoch}") if cfg.verbosity >= 2 else batches
        total = 0.0
        for begin in batches:
            batch = order[begin:begin + cfg.batch_size]
            targets = model.encode_targets(batch).to(device)
            logits = model([corrupt(w, cfg.noise, rng) for w in batch], targets)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        results = {name: spelling_metrics([s for b in range(0, len(inputs), 1024) for s in model.spell(inputs[b:b + 1024])], truth)
                   for name, (inputs, truth) in checks.items()}
        record = {"epoch": epoch, "train_loss": total / len(order), "seconds": time.time() - started}
        record.update({f"{name}_{key}": value for name, metrics in results.items() for key, value in metrics.items()})
        state["history"].append(record)
        state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     rng={"python": rng.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                     torch_version=torch.__version__, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        selection = results["unseen_noisy"]["cer"]  # the best model spells new words through noise best
        if selection < state["best_loss"]:
            state.update(best_loss=selection, best_epoch=epoch, best_model={k: v.clone() for k, v in model.state_dict().items()})
        save_atomically(state, output)
        if cfg.verbosity >= 1:
            print(f"epoch {epoch}: train loss {record['train_loss']:.4f}; accuracy / CER — "
                  + "; ".join(f"{name} {m['accuracy']:.3f}/{m['cer']:.3f}" for name, m in results.items()) + f" ({record['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
