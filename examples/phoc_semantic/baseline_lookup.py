#!/usr/bin/env python3
# Baseline for PHOC → semantics: no learning — map an input spelling to the known (training) word nearest in spelling,
# by Levenshtein or PHOC L1 distance, and use that word's vector as the prediction. It is evaluated on exactly the
# trained model's evaluation sets (same word split, same corruptions, read from its state file) with the same metric,
# so the two are directly comparable. Unseen words cannot be recovered by construction: the baseline only knows
# training words, which is precisely what the learned mapping is meant to overcome.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fargv
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from phoc import PHOC
from phoc.train_semantic import Config as TrainConfig
from phoc.train_semantic import evaluation_sets, file_digest, load_vectors, mean_metrics, rank_metrics


@dataclass
class Config:
    """Nearest-known-word lookup baseline, evaluated on a trained PHOCToSemantic's evaluation sets."""
    state: str = "data/models/phoc_semantic.pt"
    "The trained model's state file: its configuration, word split and corruption settings define the evaluation."
    distance: str = "levenshtein"
    "levenshtein | phoc"
    output: str = "baseline.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    verbosity: int = fargv.FargvInt(1, short_name="v", is_count_switch=True)


def nearest_known(inputs: List[str], known: List[str], distance: str, device: torch.device, batch_size: int) -> List[int]:
    # Index into `known` of the nearest word for every input.
    nearest: List[int] = []
    if distance == "levenshtein":
        for begin in range(0, len(inputs), batch_size):
            distances = cdist(inputs[begin:begin + batch_size], known, scorer=Levenshtein.distance, dtype=np.int32, workers=-1)
            nearest += distances.argmin(axis=1).tolist()
    elif distance == "phoc":
        phoc = PHOC().to(device)
        known_embeddings = torch.cat([phoc(known[b:b + 8192]) for b in range(0, len(known), 8192)])
        for begin in range(0, len(inputs), batch_size):
            nearest += torch.cdist(phoc(inputs[begin:begin + batch_size]), known_embeddings, p=1).argmin(dim=1).tolist()
    else:
        raise SystemExit(f"Unknown distance {distance!r}")
    return nearest


def main() -> None:
    cfg, _ = fargv.parse(Config)
    state = torch.load(cfg.state, map_location="cpu", weights_only=False)  # our own file
    train_cfg = TrainConfig(**{k: v for k, v in state["config"].items() if k in TrainConfig.__dataclass_fields__})
    if file_digest(train_cfg.vectors, train_cfg.vocabulary) != state["vectors_sha256"]:
        raise SystemExit(f"{train_cfg.vectors} is not the vectors file {cfg.state} was trained on")
    words, vectors = load_vectors(train_cfg)
    device = torch.device(cfg.device)
    row = {w: n for n, w in enumerate(words)}
    all_targets = torch.from_numpy(vectors).to(device)
    known = state["split"]["train"]
    known_rows = torch.tensor([row[w] for w in known], device=device)

    results = {"distance": cfg.distance, "state": cfg.state, "epoch_of_state": state["epoch"]}
    for name, (inputs, truth) in evaluation_sets(state["split"], train_cfg.noise, train_cfg.seed).items():
        true_rows = torch.tensor([row[w] for w in truth], device=device)
        predicted = all_targets[known_rows[torch.tensor(nearest_known(inputs, known, cfg.distance, device, cfg.batch_size), device=device)]]
        parts = [rank_metrics(predicted[b:b + cfg.batch_size], true_rows[b:b + cfg.batch_size], all_targets) for b in range(0, len(inputs), cfg.batch_size)]
        results[name] = mean_metrics(parts, len(inputs))
        if cfg.verbosity >= 1:
            print(f"{cfg.distance} lookup, {name}: top1 {results[name]['top1']:.3f} top5 {results[name]['top5']:.3f}", flush=True)
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
