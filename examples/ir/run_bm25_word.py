#!/usr/bin/env python3
# Method "bm25_word": BM25 over whole tokens; a query is the bag of its words.
import os
from dataclasses import dataclass

import fargv

from bm25_run import run_bm25, word_terms


@dataclass
class Config:
    """BM25 baseline over whole words: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    depth: int = 1000
    workers: int = os.cpu_count() or 1
    "Accepted for a uniform Makefile interface; Terrier indexes single-threaded here."


def main() -> None:
    cfg, _ = fargv.parse(Config)
    run_bm25(cfg.data_root, cfg.collection, cfg.version, cfg.output, "bm25_word", word_terms, cfg.depth)


if __name__ == "__main__":
    main()
