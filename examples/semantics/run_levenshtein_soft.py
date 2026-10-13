#!/usr/bin/env python3
# Method "levenshtein_soft": dre with edit-distance emissions — a rule word also matches its misspellings; the same
# model and defaults as the TREC-5 experiments (examples/ir/run_levenshtein_soft.py).
from dataclasses import dataclass

import fargv

from dre import CharCodeEmbedder, JointSoftmaxObserver, LevenshteinDistance
from re2rnn_rules import CLASSES
from run_common import classify


@dataclass
class Config:
    """RE2RNN rules with Levenshtein emissions: predictions for one dataset version."""
    data_root: str = "data"
    dataset: str = "ATIS"
    version: str = "clean"
    output: str = "levenshtein_soft.tsv"
    temperature: float = 0.22
    "Softness of the emission around the threshold, in edits."
    background: float = 0.27
    "Threshold in edits per character of the token."
    max_length: int = 32
    "Characters compared per token; longer tokens are truncated."
    device: str = "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    rules: str = "rules.tsv"
    "Rule file in the dataset directory (e.g. a query variation of rules.tsv)."


def main() -> None:
    cfg, _ = fargv.parse(Config)

    def make_observer(vocabulary):
        return JointSoftmaxObserver(vocabulary, CharCodeEmbedder(cfg.max_length, device=cfg.device), LevenshteinDistance(), CLASSES,
                                    temperature=cfg.temperature, background=cfg.background, device=cfg.device)

    classify(cfg.data_root, cfg.dataset, cfg.version, cfg.output, make_observer, cfg.device, cfg.engine, cfg.rules)


if __name__ == "__main__":
    main()
