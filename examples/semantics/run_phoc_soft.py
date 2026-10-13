#!/usr/bin/env python3
# Method "phoc_soft": dre with PHOC emissions — the same model and defaults as the TREC-5 experiments
# (examples/ir/run_phoc_soft.py).
from dataclasses import dataclass

import fargv

from dre import JointSoftmaxObserver, L1Distance, PHOCEmbedder
from re2rnn_rules import CLASSES
from run_common import classify


@dataclass
class Config:
    """RE2RNN rules with PHOC emissions: predictions for one dataset version."""
    data_root: str = "data"
    dataset: str = "ATIS"
    version: str = "clean"
    output: str = "phoc_soft.tsv"
    temperature: float = 2.0
    "Softness of the emission around the threshold, in PHOC L1 units."
    background: float = 0.5
    "Threshold as a fraction of the token's PHOC mass."
    device: str = "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    rules: str = "rules.tsv"
    "Rule file in the dataset directory (e.g. a query variation of rules.tsv)."


def main() -> None:
    cfg, _ = fargv.parse(Config)

    def make_observer(vocabulary):
        return JointSoftmaxObserver(vocabulary, PHOCEmbedder(device=cfg.device), L1Distance(), CLASSES,
                                    temperature=cfg.temperature, background=cfg.background, device=cfg.device)

    classify(cfg.data_root, cfg.dataset, cfg.version, cfg.output, make_observer, cfg.device, cfg.engine, cfg.rules)


if __name__ == "__main__":
    main()
