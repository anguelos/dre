#!/usr/bin/env python3
# Method "phoc_soft": the dre engine with PHOC emissions — a token's probability of being a query word falls with the
# L1 distance between their soft (counting) PHOC embeddings. The document and query vocabularies are embedded once
# per run, and their distance matrix serves every query.
import os
from dataclasses import dataclass

import fargv
import torch

from dre import JointSoftmaxObserver, L1Distance, PHOCEmbedder
from dre_run import run_dre
from phoc import PHOC


@dataclass
class Config:
    """dre with soft PHOC emissions: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    temperature: float = 2.0
    "Softness of the emission around the background threshold, in PHOC L1 units."
    background: float = 0.5
    "Background distance, as a fraction of the token's own PHOC mass: a token at that distance from a query word is as likely that word as anything else."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    workers: int = os.cpu_count() or 1
    "Accepted for a uniform Makefile interface; dre parallelises over documents on the device instead."
    verbosity: int = fargv.FargvInt(0, short_name="v", is_count_switch=True)


def main() -> None:
    cfg, _ = fargv.parse(Config)

    def make_observer(vocabulary):
        return JointSoftmaxObserver(vocabulary, PHOCEmbedder(PHOC(), device=cfg.device), L1Distance(),
                                    temperature=cfg.temperature, background=cfg.background, device=cfg.device)

    run_dre(cfg.data_root, cfg.collection, cfg.version, cfg.output, "phoc_soft", make_observer, cfg.device, verbosity=cfg.verbosity, engine=cfg.engine)


if __name__ == "__main__":
    main()
