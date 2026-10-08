#!/usr/bin/env python3
# Compile: synthetic raw/clean.jsonl → docs/{original,degrade5,degrade20}.jsonl
# The degraded versions substitute each letter with probability 5% and 20%, mimicking the TREC-5 Confusion versions.
# Each document's noise is seeded by (seed, version, docno), so the output does not depend on the number of workers.
import os
import random
import string
from functools import partial
from pathlib import Path
from typing import Tuple

import fargv

from trec_io import parallel_imap, read_docs, write_docs

NOISE = {"original": 0.0, "degrade5": 0.05, "degrade20": 0.20}


def degrade(seed: int, version: str, doc: Tuple[str, str]) -> Tuple[str, str]:
    docno, text = doc
    rng = random.Random(f"{seed}-{version}-{docno}")
    return docno, "".join(rng.choice(string.ascii_lowercase) if c.isalpha() and rng.random() < NOISE[version] else c for c in text)


def main() -> None:
    p, _ = fargv.parse({
        "data_root": "data",
        "seed": 1,
        "workers": os.cpu_count() or 1,
        "verbosity": fargv.FargvInt(0, short_name="v", is_count_switch=True),
    })
    root = Path(p.data_root) / "synthetic"
    for version in NOISE:
        docs = parallel_imap(partial(degrade, p.seed, version), read_docs(root / "raw" / "clean.jsonl"), p.workers)
        count = write_docs(root / "docs" / f"{version}.jsonl", docs)
        if p.verbosity >= 1:
            print(f"{version}: {count} documents")


if __name__ == "__main__":
    main()
