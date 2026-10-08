# Shared by the BM25 baselines: index one version's tokens with Terrier and retrieve every query with BM25.
# Documents and queries are both pre-tokenised — we hand Terrier the term counts — so its tokeniser, query parser,
# stemmer and stopword list never touch the terms. A word regex becomes a bag of its words (operators and "." are
# dropped): BM25 ignores word order and structure, which is exactly what makes it a baseline here. The index lives
# next to the run and is rebuilt with it.
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, List

import pandas as pd
import pyterrier as pt

from trec_io import Run, read_queries, read_token_strings, write_run
from word_regex import query_tokens

Terms = Callable[[List[str]], List[str]]  # tokens → index terms
# Terrier fails on terms longer than its maximum term length or with non-ASCII characters. Query terms are ASCII
# letters (plus the "_" boundary marker), so such document terms can never match; they are left out (document lengths
# shrink marginally).
MAX_TERM_LENGTH = 20


def indexable(terms: List[str]) -> List[str]:
    return [term for term in terms if term.isascii() and len(term) <= MAX_TERM_LENGTH]


def word_terms(tokens: List[str]) -> List[str]:
    return tokens


def char_ngram_terms(n: int, tokens: List[str]) -> List[str]:
    # Overlapping character n-grams of every token with its boundaries marked by "_" (McNamee & Mayfield): "register"
    # gives "_reg", "regi", …, "ter_" for n = 4, so word starts and ends are distinguished from word interiors.
    # A marked token shorter than n is kept whole.
    marked = ["_" + token + "_" for token in tokens]
    return [token[i:i + n] for token in marked for i in range(max(1, len(token) - n + 1))]


def char4_terms(tokens: List[str]) -> List[str]:
    return char_ngram_terms(4, tokens)


def char5_terms(tokens: List[str]) -> List[str]:
    return char_ngram_terms(5, tokens)


def run_bm25(data_root: str, collection: str, version: str, output: str, method: str, terms: Terms, depth: int = 1000, threads: int = 1) -> None:
    root = Path(data_root) / collection
    output_path = Path(output) if output else root / "runs" / method / f"{version}.run"
    index_path = output_path.with_suffix(".index")
    if not pt.java.started():
        pt.java.init()
    shutil.rmtree(index_path, ignore_errors=True)
    documents = ({"docno": docno, "toks": dict(Counter(indexable(terms(token_string.split()))))}
                 for docno, token_string in read_token_strings(root / "tokens" / f"{version}.tsv"))
    index = pt.IndexFactory.of(pt.IterDictIndexer(str(index_path.resolve()), meta={"docno": 32}, pretokenised=True, threads=threads).index(documents))
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=depth, properties={"termpipelines": ""}, controls={"qe": "off"})
    queries = read_queries(root / "queries.tsv")
    bags = pd.DataFrame([{"qid": q.qid, "query_toks": {term: float(count) for term, count in Counter(terms([t.lower() for t in query_tokens(q.regex) if t.isalpha()])).items()}}
                         for q in queries])
    results = bm25.transform(bags)
    run: Run = {q.qid: [] for q in queries}
    for qid, docno, score in zip(results["qid"], results["docno"], results["score"]):
        run[qid].append((docno, float(score)))
    write_run(output_path, run, tag=f"{method}_{version}", depth=depth)
