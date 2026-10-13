#!/usr/bin/env python3
# Compile: docs/<version>.jsonl → tokens/<version>.tsv, one "docno<TAB>token string" line per document.
# Tokenizing once here keeps every later step (queries, qrels, runs) on exactly the same tokens.
import os
from dataclasses import dataclass
from pathlib import Path

import fargv

from trec_io import parallel_imap, read_docs, write_token_strings
from word_regex import token_string


@dataclass
class Config:
    """Tokenize one version of a collection."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "original"
    workers: int = os.cpu_count() or 1


def tokenize(doc: tuple[str, str]) -> tuple[str, str]:
    return doc[0], token_string(doc[1])


def main() -> None:
    cfg, _ = fargv.parse(Config)
    root = Path(cfg.data_root) / cfg.collection
    docs = read_docs(root / "docs" / f"{cfg.version}.jsonl")
    write_token_strings(root / "tokens" / f"{cfg.version}.tsv", parallel_imap(tokenize, docs, cfg.workers))


if __name__ == "__main__":
    main()
