#!/usr/bin/env python3
# Method "fuzzy": the word regex matched with an edit budget by the `regex` package, on the tokens of one version.
# The budget scales with the query: k = max(1, floor(edit_rate * letters in its required words)), so short words are
# not swamped by false matches. Edits are counted on the token string and may cross word boundaries (merged or split
# words are real OCR errors). A document scores minus the edit count of its best match.
import math
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import fargv
from dre.optional import optional_import

from trec_io import Query, Run, parallel_imap, read_queries, read_token_strings, write_run
from word_regex import query_tokens, required_words, word_regex_body

regex = optional_import("regex")


@dataclass
class Config:
    """Fuzzy word-regex baseline: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    "Run file; default data_root/collection/runs/fuzzy/version.run"
    edit_rate: float = 0.2
    "Allowed edits per letter of the query's required words."
    depth: int = 1000
    workers: int = os.cpu_count() or 1


def edit_budget(query: Query, edit_rate: float) -> int:
    words = required_words(query.regex) or [token for token in query_tokens(query.regex) if token.isalpha()]
    return max(1, math.floor(edit_rate * sum(len(word) for word in words)))


def best_edits(patterns: list[tuple[str, "regex.Pattern"]], doc: tuple[str, str]) -> tuple[str, list[tuple[str, int]]]:
    found = []
    for qid, pattern in patterns:
        match = pattern.search(doc[1])
        if match is not None:
            found.append((qid, sum(match.fuzzy_counts)))
    return doc[0], found


def main() -> None:
    cfg, _ = fargv.parse(Config)
    root = Path(cfg.data_root) / cfg.collection
    output = Path(cfg.output) if cfg.output else root / "runs" / "fuzzy" / f"{cfg.version}.run"
    queries = read_queries(root / "queries.tsv")
    patterns = [(q.qid, regex.compile(r"(?<![^ ])(?:" + word_regex_body(q.regex) + r"){e<=%d}" % edit_budget(q, cfg.edit_rate), flags=regex.BESTMATCH))
                for q in queries]
    run: Run = {q.qid: [] for q in queries}
    docs = read_token_strings(root / "tokens" / f"{cfg.version}.tsv")
    for docno, found in parallel_imap(partial(best_edits, patterns), docs, cfg.workers, chunksize=16):
        for qid, edits in found:
            run[qid].append((docno, -float(edits)))
    write_run(output, run, tag=f"fuzzy_{cfg.version}", depth=cfg.depth)


if __name__ == "__main__":
    main()
