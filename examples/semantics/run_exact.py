#!/usr/bin/env python3
# Method "exact": dre with 0/1 emissions — the replication of RE2RNN's RE system (their zero-shot reference row).
from dataclasses import dataclass

import fargv

from dre import ExactObserver
from re2rnn_rules import CLASSES
from run_common import classify


@dataclass
class Config:
    """RE2RNN rules with exact matching: predictions for one dataset version."""
    data_root: str = "data"
    dataset: str = "ATIS"
    version: str = "clean"
    output: str = "exact.tsv"
    device: str = "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    rules: str = "rules.tsv"
    "Rule file in the dataset directory (e.g. a query variation of rules.tsv)."


def main() -> None:
    cfg, _ = fargv.parse(Config)

    def make_observer(vocabulary):
        return ExactObserver(vocabulary, classes=CLASSES)

    classify(cfg.data_root, cfg.dataset, cfg.version, cfg.output, make_observer, cfg.device, cfg.engine, cfg.rules)


if __name__ == "__main__":
    main()
