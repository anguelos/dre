# Train PHOCToSemantic: predict a word's semantic vector (e.g. GloVe) from its PHOC, as a denoising task — input
# spellings are corrupted with OCR-like edits, targets are the clean words' vectors. Words are split into training and
# validation sets. Validation reports how often the true word is the nearest neighbour (top-1 / top-5, by cosine) of
# the prediction among all words, for unseen words (clean and corrupted spellings: generalisation) and for fresh
# corruptions of known words (the OCR case).
#
# Everything needed to resume or reuse lives in one "fat" .pt file, rewritten atomically after every epoch: the last
# and the best model weights, optimiser state, model and training configuration, history, the word split, RNG states
# and the vectors file's checksum. Running again with the same --output resumes until --epochs is reached.
import hashlib
import os
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fargv
import numpy as np
import torch
import tqdm

from .step03_semantic import PHOCToSemantic

FORMAT = "phoc_semantic/1"
# Settings that define the model or the data; a resumed run must not change them (the vectors are checked by content).
FIXED_SETTINGS = ("hidden", "levels", "max_len", "dropout", "loss", "contrastive_temperature", "val_fraction", "max_words", "min_length", "seed")


@dataclass
class Config:
    """Train a PHOC → semantic-embedding mapping, resuming from --output if it exists."""
    vectors: str = ""
    "Word vectors: a text file 'word v1 v2 …' per line (GloVe format), or a .npy matrix with --vocabulary."
    vocabulary: str = ""
    "For a .npy --vectors: text file with the word of each row, one per line."
    output: str = "phoc_semantic.pt"
    "Fat state file; resumed if it exists."
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    hidden: str = "512,512"
    "Hidden layer sizes, comma separated."
    dropout: float = 0.1
    loss: str = "cosine"
    "cosine: 1 - cos(prediction, target). contrastive: InfoNCE — the prediction must be nearer its own word's vector than the batch's other words' vectors."
    contrastive_temperature: float = 0.05
    "Softmax temperature of the contrastive loss, in cosine units."
    levels: str = "1,2,3,5"
    max_len: int = 20
    noise: float = 0.1
    "Probability that each character of an input spelling is substituted, deleted or followed by an inserted one."
    val_fraction: float = 0.1
    "Fraction of words held out for validation."
    max_words: int = 0
    "Use only the first N words of the vectors file (0: all); GloVe files are frequency-ordered."
    min_length: int = 2
    "Ignore words shorter than this; only alphabetic words are used."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    verbosity: int = fargv.FargvInt(1, short_name="v", is_count_switch=True)
    "-v: per-epoch summary  -vv: progress bars."


def load_vectors(cfg: Config) -> Tuple[List[str], np.ndarray]:
    if cfg.vectors.endswith(".npy"):
        matrix = np.load(cfg.vectors).astype(np.float32)
        words = Path(cfg.vocabulary).read_text(encoding="utf-8").split("\n")[:len(matrix)]
    else:
        words, rows = [], []
        with open(cfg.vectors, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                words.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
        matrix = np.stack(rows)
    keep = [n for n, w in enumerate(words) if w.isalpha() and len(w) >= cfg.min_length and np.linalg.norm(matrix[n]) > 0]
    if cfg.max_words:
        keep = keep[:cfg.max_words]
    vectors = matrix[keep]
    return [words[n].lower() for n in keep], vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def file_digest(*paths: str) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def corrupt(word: str, probability: float, rng: random.Random) -> str:
    out = []
    for c in word:
        if rng.random() < probability:
            edit = rng.choice("sdi")
            if edit == "s":
                out.append(rng.choice(string.ascii_lowercase))
            elif edit == "i":
                out += [c, rng.choice(string.ascii_lowercase)]
        else:
            out.append(c)
    return "".join(out) or word


def make_split(words: List[str], val_fraction: float, seed: int) -> Dict[str, List[str]]:
    # Training and held-out words; deterministic, so every model trained with the same settings shares the split.
    order = list(range(len(words)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(val_fraction * len(words)))
    return {"train": sorted(words[n] for n in order[n_val:]), "val": sorted(words[n] for n in order[:n_val])}


def evaluation_sets(split: Dict[str, List[str]], noise: float, seed: int) -> Dict[str, Tuple[List[str], List[str]]]:
    # name → (input spellings, true words); deterministic, so the trained model and any baseline see the same inputs.
    #   unseen_clean: held-out words as spelled   unseen_noisy: held-out words, corrupted   (generalisation)
    #   known_noisy: training words in fresh corruptions                                    (the OCR case)
    unseen = split["val"]
    known = sorted(random.Random(seed).sample(split["train"], min(len(unseen), len(split["train"]))))
    return {"unseen_clean": (unseen, unseen),
            "unseen_noisy": ([corrupt(w, noise, random.Random(f"{seed}-val-{w}")) for w in unseen], unseen),
            "known_noisy": ([corrupt(w, noise, random.Random(f"{seed}-seen-{w}")) for w in known], known)}


@torch.no_grad()
def rank_metrics(predicted: torch.Tensor, true_rows: torch.Tensor, all_targets: torch.Tensor) -> Dict[str, float]:
    # predicted: (N, D) unit vectors. Sums of: cosine loss to the true word's vector, and whether the true word is the
    # top-1 / top-5 neighbour of the prediction among all words.
    ranked = (predicted @ all_targets.T).topk(5, dim=1).indices
    truth = true_rows[:, None]
    return {"loss": float((1 - (predicted * all_targets[true_rows]).sum(dim=1)).sum()),
            "top1": float((ranked[:, :1] == truth).sum()), "top5": float((ranked == truth).any(dim=1).sum())}


def mean_metrics(parts: List[Dict[str, float]], n: int) -> Dict[str, float]:
    return {key: sum(part[key] for part in parts) / n for key in ("loss", "top1", "top5")}


@torch.no_grad()
def evaluate(model: PHOCToSemantic, inputs: List[str], true_rows: torch.Tensor, all_targets: torch.Tensor, batch_size: int) -> Dict[str, float]:
    model.eval()
    parts = [rank_metrics(model(inputs[b:b + batch_size]), true_rows[b:b + batch_size], all_targets) for b in range(0, len(inputs), batch_size)]
    return mean_metrics(parts, len(inputs))


def training_loss(predicted: torch.Tensor, targets: torch.Tensor, loss: str, temperature: float) -> torch.Tensor:
    # predicted, targets: (B, D) unit vectors, row b of targets being the true vector of input b.
    if loss == "cosine":
        return (1 - (predicted * targets).sum(dim=1)).mean()
    if loss == "contrastive":
        # The other words of the batch are the negatives: pulling towards the own vector alone lets predictions drift to
        # an "average word"; pushing away from the others keeps them distinct (as in LOVE, Chen et al., ACL 2022).
        logits = predicted @ targets.T / temperature
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(predicted), device=predicted.device))
    raise SystemExit(f"Unknown loss {loss!r}: expected cosine or contrastive")


def save_atomically(state: dict, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    torch.save(state, part)
    os.replace(part, path)


def main() -> None:
    cfg, _ = fargv.parse(Config)
    if not cfg.vectors or (cfg.vectors.endswith(".npy") and not cfg.vocabulary):
        raise SystemExit("--vectors is required (with --vocabulary for a .npy matrix)")
    if cfg.loss not in ("cosine", "contrastive"):
        raise SystemExit(f"Unknown loss {cfg.loss!r}: expected cosine or contrastive")
    output = Path(cfg.output)
    vectors_digest = file_digest(cfg.vectors, cfg.vocabulary)
    words, vectors = load_vectors(cfg)
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
        state = {"format": FORMAT, "epoch": 0, "history": [], "best_loss": float("inf"), "best_epoch": 0, "vectors_sha256": vectors_digest,
                 "split": make_split(words, cfg.val_fraction, cfg.seed), "rng": None,
                 "model_config": {"output_dim": int(vectors.shape[1]), "hidden": [int(h) for h in cfg.hidden.split(",") if h],
                                  "levels": [int(level) for level in cfg.levels.split(",")], "max_len": cfg.max_len, "dropout": cfg.dropout}}
        torch.manual_seed(cfg.seed)
    state["config"] = asdict(cfg)

    model = PHOCToSemantic(**state["model_config"]).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=cfg.lr)
    if "model" in state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    rng = random.Random(cfg.seed)
    if state["rng"] is not None:
        rng.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])

    row = {w: n for n, w in enumerate(words)}
    all_targets = torch.from_numpy(vectors).to(device)
    train_words = state["split"]["train"]
    checks = {name: (inputs, torch.tensor([row[w] for w in truth], device=device))
              for name, (inputs, truth) in evaluation_sets(state["split"], cfg.noise, cfg.seed).items()}

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
            targets = all_targets[[row[w] for w in batch]]
            predicted = model([corrupt(w, cfg.noise, rng) for w in batch])
            loss = training_loss(predicted, targets, cfg.loss, cfg.contrastive_temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        results = {name: evaluate(model, inputs, rows, all_targets, cfg.batch_size) for name, (inputs, rows) in checks.items()}
        record = {"epoch": epoch, "train_loss": total / len(order), "seconds": time.time() - started}
        record.update({f"{name}_{key}": value for name, metrics in results.items() for key, value in metrics.items()})
        state["history"].append(record)
        clean = results["unseen_clean"]
        state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     rng={"python": rng.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                     torch_version=torch.__version__, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        if clean["loss"] < state["best_loss"]:
            state.update(best_loss=clean["loss"], best_epoch=epoch, best_model={k: v.clone() for k, v in model.state_dict().items()})
        save_atomically(state, output)
        if cfg.verbosity >= 1:
            print(f"epoch {epoch}: train loss {record['train_loss']:.4f}; top1/top5 — "
                  + "; ".join(f"{name} {m['top1']:.3f}/{m['top5']:.3f}" for name, m in results.items()) + f" ({record['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
