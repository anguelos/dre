#!/usr/bin/env python3
# Compile: queries.tsv + tokens/<reference>.tsv → qrels.txt
# A document is relevant to a query iff the query matches its reference (clean) text exactly. Every document is
# tested, so the qrels are complete. Queries with no relevant document, or with more than max_relevant (more than a
# run can list, so recall and AP are meaningless), are left out of qrels.txt and listed in qrels.dropped.tsv.
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Tuple

import fargv

from trec_io import Qrels, atomic_write, parallel_imap, read_queries, read_token_strings, write_qrels
from word_regex import compile_word_regex, may_match, required_words


@dataclass
class Config:
    """Exhaustive qrels: run every query exactly on the reference version."""
    data_root: str = "data"
    collection: str = "trec5"
    reference: str = "original"
    workers: int = os.cpu_count() or 1
    max_relevant: int = 1000
    "Queries with more relevant documents than this are dropped; match it to the run depth."


def matching_queries(patterns: List[Tuple[str, re.Pattern, List[str]]], doc: Tuple[str, str]) -> Tuple[str, List[str]]:
    padded = " " + doc[1]
    return doc[0], [qid for qid, pattern, required in patterns if may_match(required, padded) and pattern.search(doc[1])]


def main() -> None:
    cfg, _ = fargv.parse(Config)
    root = Path(cfg.data_root) / cfg.collection
    queries = read_queries(root / "queries.tsv")
    patterns = [(query.qid, compile_word_regex(query.regex), required_words(query.regex)) for query in queries]
    qrels: Qrels = {}
    docs = read_token_strings(root / "tokens" / f"{cfg.reference}.tsv")
    for docno, qids in parallel_imap(partial(matching_queries, patterns), docs, cfg.workers):
        for qid in qids:
            qrels.setdefault(qid, {})[docno] = 1
    dropped = [(query, len(qrels.get(query.qid, {}))) for query in queries if not 0 < len(qrels.get(query.qid, {})) <= cfg.max_relevant]
    with atomic_write(root / "qrels.dropped.tsv") as f:
        for query, n_relevant in dropped:
            f.write(f"{query.qid}\t{query.family}\t{n_relevant}\t{query.regex}\n")
            qrels.pop(query.qid, None)
    write_qrels(root / "qrels.txt", qrels)
    print(f"{len(qrels)} queries evaluated, {len(dropped)} dropped (no relevant document, or more than {cfg.max_relevant})")


if __name__ == "__main__":
    main()
