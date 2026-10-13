#!/usr/bin/env python3
# Evaluate: one run + qrels → per-query scores and per-family means, as a TSV (the input for significance tests).
# Two deliberate choices, both stated in the paper's protocol:
#   - documents are scored by their rank in the run file, so tie order is the one the run declares; trec_eval-style
#     tools would otherwise re-break score ties by docno in reverse order;
#   - every query in the qrels is evaluated: a query the run leaves empty scores 0 instead of silently disappearing.
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import fargv
from dre.optional import optional_import

from trec_io import atomic_write, read_qrels, read_queries, read_run

ir_measures = optional_import("ir_measures")


def measures() -> list:
    return [ir_measures.AP, ir_measures.nDCG @ 10, ir_measures.P @ 10, ir_measures.R @ 1000, ir_measures.RR]


@dataclass
class Config:
    """Evaluate one TREC run file against qrels."""
    qrels: str = "data/trec5/qrels.txt"
    queries: str = "data/trec5/queries.tsv"
    run: str = ""
    output: str = ""
    "TSV with one row per query, then one mean row per family and over all queries."
    verbosity: int = fargv.FargvInt(0, short_name="v", is_count_switch=True)
    "-v: print the means."


def main() -> None:
    cfg, _ = fargv.parse(Config)
    qrels = read_qrels(cfg.qrels)
    families = {query.qid: query.family for query in read_queries(cfg.queries)}
    run = {qid: {docno: -float(rank) for rank, (docno, _) in enumerate(ranked, start=1)} for qid, ranked in read_run(cfg.run).items() if qid in qrels}
    scores: dict[str, dict[str, float]] = {qid: {str(m): 0.0 for m in measures()} for qid in qrels}
    for metric in ir_measures.iter_calc(measures(), qrels, run):
        scores[metric.query_id][str(metric.measure)] = metric.value
    names = [str(m) for m in measures()]
    groups: dict[str, list[str]] = {}
    for qid in sorted(scores):
        groups.setdefault(families[qid], []).append(qid)
    groups["all"] = sorted(scores)
    with atomic_write(cfg.output) as f:
        f.write("qid\tfamily\t" + "\t".join(names) + "\n")
        for qid in sorted(scores):
            f.write(f"{qid}\t{families[qid]}\t" + "\t".join(f"{scores[qid][n]:.4f}" for n in names) + "\n")
        for group, qids in groups.items():
            f.write(f"mean\t{group}\t" + "\t".join(f"{mean(scores[q][n] for q in qids):.4f}" for n in names) + "\n")
    if cfg.verbosity >= 1:
        print(f"{Path(cfg.run).name}: " + "  ".join(f"{n} {mean(scores[q][n] for q in groups['all']):.3f}" for n in names))


if __name__ == "__main__":
    main()
