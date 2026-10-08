#!/usr/bin/env python3
# Compile: tokens/<reference>.tsv → queries.tsv, three query families sampled reproducibly from the reference text:
#   word    — single words, sampled uniformly within document-frequency bands
#   phrase  — 2- and 3-word phrases occurring in the text
#   pattern — 3-word phrases "a b c" rewritten as "a . c", "a b ? c" or "a ( b | x ) c", x a random vocabulary word
# Every query is derived from the text itself, so none is hand-picked and each matches at least one document.
# A 3-gram "a b c" is only accepted if the rarer of a, b and the rarer of a, c both fall within the word bands: the
# words every template keeps must include one that is not a function word, or "to the" would match everything.
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import fargv

from trec_io import Query, read_token_strings, write_queries


@dataclass
class Config:
    """Sample word, phrase and pattern queries from the reference version of a collection."""
    data_root: str = "data"
    collection: str = "trec5"
    reference: str = "original"
    bands: str = "2-10,11-100,101-1000"
    "Document-frequency bands for word queries, as lo-hi ranges; their overall range also filters phrase words."
    per_band: int = 20
    phrases: int = 30
    patterns: int = 30
    min_word_len: int = 3
    "Minimum length of sampled single-word queries."
    seed: int = 1
    verbosity: int = fargv.FargvInt(0, short_name="v", is_count_switch=True)
    "-v: report counts."


def main() -> None:
    cfg, _ = fargv.parse(Config)
    root = Path(cfg.data_root) / cfg.collection
    rng = random.Random(cfg.seed)
    bands = [tuple(int(v) for v in band.split("-")) for band in cfg.bands.split(",")]
    lowest, highest = min(lo for lo, _ in bands), max(hi for _, hi in bands)
    tokens_path = root / "tokens" / f"{cfg.reference}.tsv"
    document_frequency: Counter = Counter()
    for _, token_string in read_token_strings(tokens_path):
        document_frequency.update(set(token_string.split()))

    def in_bands(*words: str) -> bool:
        return lowest <= min(document_frequency[w] for w in words) <= highest

    n_ngrams = cfg.phrases + cfg.patterns
    reservoir: List[List[str]] = []
    n_candidates = 0
    for _, token_string in read_token_strings(tokens_path):
        tokens = token_string.split()
        # Tokens are already lower-cased, so ASCII-alphabetic is exactly the query language's [a-z]+.
        flags = [token.isascii() and token.isalpha() for token in tokens]
        starts = [i for i in range(len(tokens) - 2) if flags[i] and flags[i + 1] and flags[i + 2]
                  and in_bands(tokens[i], tokens[i + 1]) and in_bands(tokens[i], tokens[i + 2])]
        if not starts:
            continue
        ngram = tokens[rng.choice(starts):][:3]
        # Reservoir sampling: one 3-gram per document, uniform over documents, in a single deterministic pass.
        if n_candidates < n_ngrams:
            reservoir.append(ngram)
        else:
            j = rng.randrange(n_candidates + 1)
            if j < n_ngrams:
                reservoir[j] = ngram
        n_candidates += 1

    vocabulary = sorted(w for w in document_frequency if re.fullmatch(r"[a-z]+", w))
    regexes: List[Tuple[str, str]] = []
    for lo, hi in bands:
        candidates = [w for w in vocabulary if lo <= document_frequency[w] <= hi and len(w) >= cfg.min_word_len]
        regexes += [("word", w) for w in rng.sample(candidates, min(cfg.per_band, len(candidates)))]
    for n, (a, b, c) in enumerate(reservoir[:cfg.phrases]):
        regexes.append(("phrase", f"{a} {b}" if n % 2 == 0 else f"{a} {b} {c}"))
    for n, (a, b, c) in enumerate(reservoir[cfg.phrases:]):
        templates = (f"{a} . {c}", f"{a} {b} ? {c}", f"{a} ( {b} | {rng.choice(vocabulary)} ) {c}")
        regexes.append(("pattern", templates[n % 3]))

    seen = set()
    queries = []
    for family, regex in regexes:
        if regex not in seen:
            seen.add(regex)
            queries.append(Query(f"q{len(queries) + 1:04d}", family, regex))
    write_queries(root / "queries.tsv", queries)
    if cfg.verbosity >= 1:
        print(f"{len(queries)} queries: " + ", ".join(f"{f} {c}" for f, c in Counter(q.family for q in queries).items()))


if __name__ == "__main__":
    main()
