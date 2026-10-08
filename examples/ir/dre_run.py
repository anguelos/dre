# Shared by the dre methods: load a version's corpus and the queries, prepare the observer once for the whole
# corpus vocabulary and all query words, then score all documents for every query and write the run file. Only the
# observer differs between methods.
import time
from pathlib import Path
from typing import Callable

import torch

from dre import DFAMatcher, Searcher, SequenceObserver, ViterbiMatcher, Vocabulary, compile_regex, query_words
from trec_io import Run, read_corpus, read_queries, write_run

# Given the document vocabulary, returns the observer of a method.
ObserverFactory = Callable[[Vocabulary], SequenceObserver]
MATCHERS = {"viterbi": ViterbiMatcher, "dfa": DFAMatcher}


def run_dre(data_root: str, collection: str, version: str, output: str, method: str, make_observer: ObserverFactory, device: str,
            depth: int = 1000, verbosity: int = 0, engine: str = "viterbi") -> None:
    root = Path(data_root) / collection
    output_path = Path(output) if output else root / "runs" / method / f"{version}.run"
    started = time.time()
    docnos, corpus = read_corpus(root / "tokens" / f"{version}.tsv")
    queries = read_queries(root / "queries.tsv")
    automata = [compile_regex(query.regex) for query in queries]
    words = sorted({word for automaton in automata for word in query_words(automaton)})
    if verbosity >= 1:
        print(f"{version}: {len(corpus)} documents, {len(corpus.vocabulary)} distinct tokens, {len(words)} query words; "
              f"loaded in {time.time() - started:.0f} s", flush=True)
    searcher = Searcher(corpus, make_observer(corpus.vocabulary), MATCHERS[engine](device=torch.device(device)))
    searcher.observer.prepare(words)
    if verbosity >= 1:
        print(f"observer prepared at {time.time() - started:.0f} s", flush=True)
    run: Run = {}
    for query, automaton in zip(queries, automata):
        scores = searcher.scores(automaton)
        matched = scores.nonzero().reshape(-1).tolist()
        run[query.qid] = [(docnos[n], float(scores[n])) for n in matched]
        if verbosity >= 2:
            print(f"{query.qid} {query.regex!r}: {len(matched)} documents with a non-zero score, at {time.time() - started:.0f} s", flush=True)
    write_run(output_path, run, tag=f"{method}_{version}", depth=depth)
    if verbosity >= 1:
        print(f"{method} on {version}: {time.time() - started:.0f} s", flush=True)
