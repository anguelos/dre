#!/usr/bin/env python3
# Method "dre_exact": the dre engine with 0/1 emissions (a token is a query word iff it is that word).
# It must retrieve exactly the documents the `re` baseline retrieves; this checks the compiler and forward pass.
import os
from dataclasses import dataclass

import fargv
import torch

from dre import ExactObserver
from dre_run import run_dre


@dataclass
class Config:
    """dre with exact emissions: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    workers: int = os.cpu_count() or 1
    "Accepted for a uniform Makefile interface; dre parallelises over documents on the device instead."
    verbosity: int = fargv.FargvInt(0, short_name="v", is_count_switch=True)


def main() -> None:
    cfg, _ = fargv.parse(Config)
    run_dre(cfg.data_root, cfg.collection, cfg.version, cfg.output, "dre_exact", ExactObserver, cfg.device, verbosity=cfg.verbosity, engine=cfg.engine)


if __name__ == "__main__":
    main()
