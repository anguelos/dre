#!/usr/bin/env python3
# Evaluate: one run + qrels → per-query scores and per-family means, as a TSV (the input for significance tests).
# Two deliberate choices, both stated in the paper's protocol:
#   - documents are scored by their rank in the run file, so tie order is the one the run declares; trec_eval-style
#     tools would otherwise re-break score ties by docno in reverse order;
#   - every query in the qrels is evaluated: a query the run leaves empty scores 0 instead of silently disappearing.
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List

import fargv
import ir_measures
from ir_measures import AP, RR, P, R, nDCG

from trec_io import atomic_write, read_qrels, read_queries, read_run

MEASURES = [AP, nDCG @ 10, P @ 10, R @ 1000, RR]


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
    scores: Dict[str, Dict[str, float]] = {qid: {str(m): 0.0 for m in MEASURES} for qid in qrels}
    for metric in ir_measures.iter_calc(MEASURES, qrels, run):
        scores[metric.query_id][str(metric.measure)] = metric.value
    names = [str(m) for m in MEASURES]
    groups: Dict[str, List[str]] = {}
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
