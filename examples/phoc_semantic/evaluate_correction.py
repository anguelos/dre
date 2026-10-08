#!/usr/bin/env python3
# Task A — spelling (OCR) correction: recover a word's clean spelling from a corrupted one. Evaluated on the same
# evaluation sets as the semantic models (same split, same corruptions), reporting word accuracy and character error
# rate (CER) separately for known words (in the dictionary of training words) and unseen words (not in it):
#   identity              the input unchanged (the floor)
#   lookup, Levenshtein   the dictionary word nearest in edit distance
#   lookup, PHOC          the dictionary word nearest in PHOC L1 distance
#   noisy channel         argmax over dictionary words w of log P(w) - edit_cost * edits(input, w): a Zipf prior from
#                         the frequency-ordered vectors file (P(w) ∝ 1 / rank) and a fixed cost per edit (Norvig-style)
#   decoder <name>        PHOCToString: spells whatever the input's PHOC encodes, with no dictionary
# Dictionary methods cannot produce unseen words by construction; that is where a decoder can differ.
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import fargv
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from phoc import PHOC, PHOCToString
from phoc.train_semantic import Config as TrainConfig
from phoc.train_semantic import evaluation_sets, load_vectors

Corrector = Callable[[List[str]], List[str]]


@dataclass
class Config:
    """Evaluate spelling correction on the trained models' evaluation sets."""
    decoders: str = "data/models/phoc_decoder.*.pt"
    "Trained PHOCToString state files (glob); the first one's split and corruption settings define the evaluation."
    edit_cost: float = 6.7
    "Noisy-channel cost per edit, in nats, from the training noise model: a substitution or insertion has probability 0.1 * 1/3 * 1/26 ≈ 0.0013, and -ln(0.0013) ≈ 6.7."
    output: str = "data/results/correction.json"
    batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def cer(predicted: List[str], truth: List[str]) -> float:
    return sum(Levenshtein.distance(p, t) for p, t in zip(predicted, truth)) / sum(len(t) for t in truth)


def main() -> None:
    cfg, _ = fargv.parse(Config)
    decoder_paths = sorted(glob.glob(cfg.decoders))
    first = torch.load(decoder_paths[0], map_location="cpu", weights_only=False)  # our own file
    train_cfg = TrainConfig(**{k: v for k, v in first["config"].items() if k in TrainConfig.__dataclass_fields__})
    words, _ = load_vectors(train_cfg)
    dictionary = first["split"]["train"]
    rank = {w: n for n, w in enumerate(words)}  # vectors files are frequency-ordered
    log_prior = -np.log(1 + np.array([rank[w] for w in dictionary], dtype=np.float64))
    device = torch.device(cfg.device)

    def by_distance(inputs: List[str], score: Callable[[np.ndarray], np.ndarray]) -> List[str]:
        out = []
        for b in range(0, len(inputs), cfg.batch_size):
            edits = cdist(inputs[b:b + cfg.batch_size], dictionary, scorer=Levenshtein.distance, dtype=np.int32, workers=-1)
            out += [dictionary[n] for n in score(edits).argmax(axis=1)]
        return out

    def lookup_phoc(inputs: List[str]) -> List[str]:
        phoc = PHOC().to(device)
        reference = torch.cat([phoc(dictionary[b:b + 8192]) for b in range(0, len(dictionary), 8192)])
        nearest = torch.cat([torch.cdist(phoc(inputs[b:b + cfg.batch_size]), reference, p=1).argmin(dim=1).cpu() for b in range(0, len(inputs), cfg.batch_size)])
        return [dictionary[int(n)] for n in nearest]

    def decoder(path: str) -> Corrector:
        def correct(inputs: List[str]) -> List[str]:
            model = PHOCToString.from_checkpoint(path).to(device)
            return [s for b in range(0, len(inputs), cfg.batch_size) for s in model.spell(inputs[b:b + cfg.batch_size])]
        return correct

    methods: Dict[str, Corrector] = {
        "identity (input unchanged)": lambda inputs: list(inputs),
        "lookup: nearest dictionary word, Levenshtein": lambda inputs: by_distance(inputs, lambda edits: -edits),
        "lookup: nearest dictionary word, PHOC": lookup_phoc,
        f"noisy channel: Zipf prior, {cfg.edit_cost} nats per edit": lambda inputs: by_distance(inputs, lambda edits: log_prior[None] - cfg.edit_cost * edits),
    }
    for path in decoder_paths:
        methods[f"decoder {Path(path).stem.split('.')[-1]}"] = decoder(path)

    sets = evaluation_sets(first["split"], train_cfg.noise, train_cfg.seed)
    results: Dict = {}
    for method, correct in methods.items():
        results[method] = {}
        for name, (inputs, truth) in sets.items():
            predicted = correct(inputs)
            results[method][name] = {"accuracy": float(np.mean([p == t for p, t in zip(predicted, truth)])), "cer": cer(predicted, truth),
                                     "examples": [[i, p, t] for i, p, t in list(zip(inputs, predicted, truth))[:8]]}
        print(method, {k: f"{v['accuracy']:.3f}/{v['cer']:.3f}" for k, v in results[method].items()}, flush=True)
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
