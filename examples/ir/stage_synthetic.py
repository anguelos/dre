#!/usr/bin/env python3
# Stage: generate a small synthetic collection (no internet) in <data_root>/synthetic/raw/, for fast end-to-end tests.
# Documents are word sequences drawn from a Zipf-distributed vocabulary of pronounceable pseudo-words.
import hashlib
import random
from pathlib import Path

import fargv

from trec_io import atomic_write, write_docs


def main() -> None:
    p, _ = fargv.parse({
        "data_root": "data",
        "nb_docs": 200,
        "seed": 1,
    })
    rng = random.Random(p.seed)
    syllables = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
    vocabulary = sorted({"".join(rng.choice(syllables) for _ in range(rng.randint(1, 4))) for _ in range(400)})
    rng.shuffle(vocabulary)
    weights = [1.0 / (rank + 1) for rank in range(len(vocabulary))]
    docs = []
    for n in range(p.nb_docs):
        words = rng.choices(vocabulary, weights=weights, k=rng.randint(40, 160))
        text = " ".join(word + ("." if rng.random() < 0.08 else "") for word in words)
        docs.append((f"SYN-{n:05d}", text))
    raw = Path(p.data_root) / "synthetic" / "raw"
    write_docs(raw / "clean.jsonl", docs)
    digest = hashlib.sha256((raw / "clean.jsonl").read_bytes()).hexdigest()
    with atomic_write(raw / "SHA256SUMS") as f:
        f.write(f"{digest}  clean.jsonl\n")


if __name__ == "__main__":
    main()
