#!/usr/bin/env python3
# One method on one (benchmark, dataset, document version), for every query variant made by augment.py that has no
# result yet: data/results/<benchmark>/<method>/<dataset>.<version>.<variant>.json, plus the predictions (re2rnn) or
# the run file (trec5) under data/runs/. Methods: dre ones as <method>.<engine> (matching.py: exact, phoc,
# levenshtein, cosine<dim>, csls<dim>, phoc_csls<dim>; viterbi or dfa), baselines (baselines.py): re, fuzzy,
# bm25_word, bm25_char4 (bm25 and re: trec5 only).
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fargv
import torch

import baselines
import benchmarks
from matching import observer_factory, split_method


@dataclass
class Config:
    """Run one method on all query variants of one benchmark dataset and document version."""
    benchmark: str = "re2rnn"
    "re2rnn or trec5"
    method: str = "phoc.viterbi"
    dataset: str = "ATIS"
    "re2rnn: ATIS, TREC (QC) or SMS; trec5: trec5"
    version: str = "clean"
    "re2rnn: clean, noise5, noise20; trec5: original, degrade5, degrade20"
    variants: str = ""
    "Space-separated variants (e.g. 'original synonym1'); empty: every variant augment.py wrote."
    data_root: str = "data"
    re2rnn_root: str = "../semantics/data"
    trec_root: str = "../ir/data/trec5"
    glove_root: str = "data/glove"
    "glove<dim>.pt from stage.py; the CSLS radii are cached there too."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = os.cpu_count() or 1
    "Processes for the re, fuzzy and bm25 baselines."
    force: bool = False
    verbosity: int = fargv.FargvInt(1, short_name="v", is_count_switch=True)


def variant_files(cfg: Config) -> dict[str, Path]:
    root = Path(cfg.data_root) / cfg.benchmark / (cfg.dataset if cfg.benchmark == "re2rnn" else "")
    prefix = "rules" if cfg.benchmark == "re2rnn" else "queries"
    found = {p.name[len(prefix) + 1:-len(".tsv")]: p for p in sorted(root.glob(f"{prefix}.*.tsv")) if not p.name.endswith(".log.tsv")}
    wanted = cfg.variants.split() or list(found)
    return {v: found[v] for v in wanted}


def result_path(cfg: Config, variant: str) -> Path:
    return Path(cfg.data_root) / "results" / cfg.benchmark / cfg.method / f"{cfg.dataset}.{cfg.version}.{variant}.json"


def run_re2rnn(cfg: Config, todo: dict[str, Path], say) -> None:
    labels, gold, sentences = benchmarks.read_sentences(cfg.re2rnn_root, cfg.dataset, cfg.version)
    rule_sets = {v: benchmarks.read_rules(p) for v, p in todo.items()}
    if cfg.method == "fuzzy":
        scorer = lambda variant: baselines.fuzzy_rule_scores(sentences, [rule for _, rule in rule_sets[variant]])
    else:
        method, engine = split_method(cfg.method)
        make_observer = observer_factory(method, benchmarks.CLASSES, cfg.glove_root, cfg.device)
        corpus = benchmarks.dre_corpus([" ".join(s) for s in sentences], make_observer, engine, True, cfg.device)
        automata = dict(zip(rule_sets, benchmarks.dre_prepare(corpus, [[r for _, r in rules] for rules in rule_sets.values()])))
        say("observer prepared")
        scorer = lambda variant: (torch.stack([corpus.scores(a) for a in automata[variant]], dim=1).tolist(), 0)
    for variant, rules in rule_sets.items():
        scores, timeouts = scorer(variant)
        result, predicted = benchmarks.re2rnn_result(scores, rules, labels, gold, cfg.dataset)
        result["timeouts"] = timeouts
        out = Path(cfg.data_root) / "runs" / cfg.benchmark / cfg.method / f"{cfg.dataset}.{cfg.version}.{variant}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{g}\t{p}\n" for g, p in zip(gold, predicted)))
        benchmarks.write_json(result_path(cfg, variant), result)
        say(f"{variant}: accuracy {100 * result['accuracy']:.2f}")


def run_trec5(cfg: Config, todo: dict[str, Path], say) -> None:
    qrels = benchmarks.read_qrels(cfg.trec_root)
    query_sets = {v: benchmarks.read_queries(p) for v, p in todo.items()}
    runs: dict[str, dict[str, list[str]]] = {v: {} for v in query_sets}
    if cfg.method in ("re", "fuzzy") or cfg.method.startswith("bm25"):
        keyed = [(f"{v}\t{qid}", regex) for v, queries in query_sets.items() for qid, _, regex in queries]
        docs = benchmarks.read_tokens(cfg.trec_root, cfg.version)
        if cfg.method.startswith("bm25"):
            scores = baselines.bm25_document_scores(cfg.method, docs, keyed, cfg.workers)
        else:
            scores = baselines.regex_document_scores(cfg.method, docs, keyed, cfg.workers)
        for key, by_doc in scores.items():
            variant, qid = key.split("\t")
            runs[variant][qid] = benchmarks.rank(by_doc)
    else:
        method, engine = split_method(cfg.method)
        make_observer = observer_factory(method, None, cfg.glove_root, cfg.device)
        documents = list(benchmarks.read_tokens(cfg.trec_root, cfg.version))
        docnos = [docno for docno, _ in documents]
        corpus = benchmarks.dre_corpus([tokens for _, tokens in documents], make_observer, engine, False, cfg.device)
        del documents
        say(f"corpus: {len(docnos)} documents, {len(corpus.vocabulary)} distinct tokens")
        automata = dict(zip(query_sets, benchmarks.dre_prepare(corpus, [[rx for _, _, rx in qs] for qs in query_sets.values()])))
        say("observer prepared")
        for variant, queries in query_sets.items():
            for (qid, _, _), automaton in zip(queries, automata[variant]):
                scores = corpus.scores(automaton)
                matched = scores.nonzero().reshape(-1).tolist()
                runs[variant][qid] = benchmarks.rank({docnos[n]: float(scores[n]) for n in matched})
            write_trec5(cfg, variant, queries, runs[variant], qrels, say)  # each variant as soon as it is done
        return
    for variant, queries in query_sets.items():
        write_trec5(cfg, variant, queries, runs[variant], qrels, say)


def write_trec5(cfg: Config, variant: str, queries, run: dict[str, list[str]], qrels, say) -> None:
    result = benchmarks.trec5_result(run, queries, qrels)
    benchmarks.write_run(Path(cfg.data_root) / "runs" / cfg.benchmark / cfg.method / f"{cfg.version}.{variant}.run",
                         run, f"{cfg.method}_{cfg.version}_{variant}")
    benchmarks.write_json(result_path(cfg, variant), result)
    say(f"{variant}: MAP {result['means']['all']['AP']:.4f}")
    subprocess.run([sys.executable, "report.py", f"--data_root={cfg.data_root}"], check=False)  # REPORT.md now, not after all variants


def main() -> None:
    cfg, _ = fargv.parse(Config)
    started = time.time()

    def say(message: str) -> None:
        if cfg.verbosity:
            print(f"[{time.time() - started:6.0f} s] {cfg.benchmark} {cfg.method} {cfg.dataset} {cfg.version}: {message}", flush=True)

    todo = {v: p for v, p in variant_files(cfg).items() if cfg.force or not result_path(cfg, v).exists()}
    if not todo:
        say("nothing to do")
        return
    say(f"{len(todo)} variants")
    (run_re2rnn if cfg.benchmark == "re2rnn" else run_trec5)(cfg, todo, say)
    say("done")


if __name__ == "__main__":
    main()
