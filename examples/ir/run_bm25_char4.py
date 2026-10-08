#!/usr/bin/env python3
# Method "bm25_char4": BM25 over overlapping character 4-grams of every token, word boundaries marked with "_" —
# the classic OCR-robust baseline (McNamee & Mayfield; TREC-5 Confusion track); a query is the bag of its words' 4-grams.
import os
from dataclasses import dataclass

import fargv

from bm25_run import char4_terms, run_bm25


@dataclass
class Config:
    """BM25 baseline over character 4-grams with word-boundary markers: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    depth: int = 1000
    workers: int = os.cpu_count() or 1
    "Accepted for a uniform Makefile interface; Terrier indexes single-threaded here."


def main() -> None:
    cfg, _ = fargv.parse(Config)
    run_bm25(cfg.data_root, cfg.collection, cfg.version, cfg.output, "bm25_char4", char4_terms, cfg.depth)


if __name__ == "__main__":
    main()
