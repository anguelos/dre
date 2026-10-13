#!/usr/bin/env python3
# Method "glove_soft": dre with semantic emissions — the distance between a token and a rule word is the cosine
# distance of their GloVe vectors (RE2RNN's own 100-d matrices), so a rule word also matches its near-synonyms; this is
# the zero-shot counterpart of FA-RNN's GloVe relaxation. Tokens without a vector (e.g. OCR-corrupted ones) match
# only themselves. A token's mass is 1, so `background` is a plain cosine-distance threshold.
from dataclasses import dataclass
from pathlib import Path

import fargv
import numpy as np
import torch

from dre import CosineDistance, JointSoftmaxObserver, TableEmbedder
from re2rnn_rules import CLASSES
from run_common import classify


@dataclass
class Config:
    """RE2RNN rules with GloVe-similarity emissions: predictions for one dataset version."""
    data_root: str = "data"
    dataset: str = "ATIS"
    version: str = "clean"
    output: str = "glove_soft.tsv"
    temperature: float = 0.05
    "Softness of the emission around the threshold, in cosine-distance units."
    background: float = 0.4
    "Cosine distance at which a token is as likely the rule word as anything else."
    device: str = "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    rules: str = "rules.tsv"
    "Rule file in the dataset directory (e.g. a query variation of rules.tsv)."


def main() -> None:
    cfg, _ = fargv.parse(Config)
    root = Path(cfg.data_root) / cfg.dataset
    vectors = np.load(root / "glove.npy")
    words = (root / "glove_vocab.txt").read_text().split("\n")[:len(vectors)]
    embedder = TableEmbedder(words, torch.from_numpy(vectors.astype(np.float32)), device=cfg.device)  # no vector: zeros

    def make_observer(vocabulary):
        return JointSoftmaxObserver(vocabulary, embedder, CosineDistance(), CLASSES,
                                    temperature=cfg.temperature, background=cfg.background, device=cfg.device)

    classify(cfg.data_root, cfg.dataset, cfg.version, cfg.output, make_observer, cfg.device, cfg.engine, cfg.rules)


if __name__ == "__main__":
    main()
