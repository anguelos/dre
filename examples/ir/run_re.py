#!/usr/bin/env python3
# Method "re": exact word-regex matching on the tokens of one version, scored by the number of matches.
# On the reference version this reproduces the qrels exactly, a sanity check of the whole pipeline.
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Tuple

import fargv

from trec_io import Run, parallel_imap, read_queries, read_token_strings, write_run
from word_regex import compile_word_regex, may_match, required_words


@dataclass
class Config:
    """Exact word-regex baseline: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    "Run file; default data_root/collection/runs/re/version.run"
    depth: int = 1000
    workers: int = os.cpu_count() or 1


def count_matches(patterns: List[Tuple[str, re.Pattern, List[str]]], doc: Tuple[str, str]) -> Tuple[str, List[Tuple[str, int]]]:
    padded = " " + doc[1]
    counts = [(qid, sum(1 for _ in pattern.finditer(doc[1]))) for qid, pattern, required in patterns if may_match(required, padded)]
    return doc[0], [(qid, count) for qid, count in counts if count]


def main() -> None:
    cfg, _ = fargv.parse(Config)
    root = Path(cfg.data_root) / cfg.collection
    output = Path(cfg.output) if cfg.output else root / "runs" / "re" / f"{cfg.version}.run"
    patterns = [(query.qid, compile_word_regex(query.regex), required_words(query.regex)) for query in read_queries(root / "queries.tsv")]
    run: Run = {qid: [] for qid, _, _ in patterns}
    docs = read_token_strings(root / "tokens" / f"{cfg.version}.tsv")
    for docno, counts in parallel_imap(partial(count_matches, patterns), docs, cfg.workers):
        for qid, count in counts:
            run[qid].append((docno, float(count)))
    write_run(output, run, tag=f"re_{cfg.version}", depth=cfg.depth)


if __name__ == "__main__":
    main()
