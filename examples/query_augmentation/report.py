#!/usr/bin/env python3
# REPORT.md from whatever results exist (data/results), rewritten atomically, so it can run after every result.
# Per benchmark, dataset and document version: methods as rows (published / baseline rows first), query variants as
# columns: the original rules/queries, then every generator on its own: mean ± std over its seeds and the relative
# change to the original, 100 * (mean - original) / original. "…" marks results still pending.
import json
import os
import time
from pathlib import Path
from statistics import mean, pstdev

import fargv

from augment import GENERATORS

# published zero-shot accuracy with the original rules on clean test sets (Jiang et al., EMNLP 2020)
PUBLISHED = {"RE (published)": {"ATIS": 87.01, "TREC": 64.40, "SMS": 93.20},
             "FA-RNN (published)": {"ATIS": 86.53, "TREC": 61.95, "SMS": 93.00},
             "BiFA-GRU (published)": {"ATIS": 88.63, "TREC": 62.90, "SMS": 93.20}}
DATASET_NAMES = {"ATIS": "ATIS", "TREC": "QC (TREC questions)", "SMS": "SMS spam", "trec5": "TREC-5 Confusion track"}
VERSION_NAMES = {"clean": "clean text", "noise5": "5% character noise", "noise20": "20% character noise",
                 "original": "original documents", "degrade5": "OCR, 5% character errors", "degrade20": "OCR, 20% character errors"}
GRID = {
    "re2rnn": {"datasets": ["ATIS", "TREC", "SMS"], "versions": ["clean", "noise5", "noise20"], "seeds": [1, 2, 3],
               "methods": ["exact.viterbi", "levenshtein.viterbi", "phoc.viterbi", "cosine100.viterbi",
                           "csls100.viterbi", "phoc_csls100.viterbi",
                           "exact.dfa", "levenshtein.dfa", "phoc.dfa", "csls100.dfa", "phoc_csls100.dfa"]},
    "trec5": {"datasets": ["trec5"], "versions": ["original", "degrade5", "degrade20"], "seeds": [1],
              "methods": ["bm25_word", "bm25_char4", "re", "fuzzy", "exact.viterbi", "levenshtein.viterbi", "phoc.viterbi",
                          "phoc_csls100.viterbi"]},
}
ONLY_VERSIONS = {("trec5", "fuzzy"): ["original"]}  # fuzzy takes ~2 h per pass over TREC-5


def score(path: Path, benchmark: str) -> float | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    return 100 * result["accuracy"] if benchmark == "re2rnn" else result["means"]["all"]["AP"]


def cell(original: float | None, values: list[float], seeds: int, digits: int) -> str:
    if not values:
        return "…"
    text = f"{mean(values):.{digits}f}" + (f" ± {pstdev(values):.{digits}f}" if len(values) > 1 else "")
    if original:
        text += f" ({100 * (mean(values) - original) / original:+.1f}%)"
    return text + ("" if len(values) == seeds else f" [{len(values)}/{seeds}]")


def table(root: Path, benchmark: str, dataset: str, version: str) -> tuple[list[str], int, int]:
    grid, digits = GRID[benchmark], 2 if benchmark == "re2rnn" else 4
    seeds = grid["seeds"]
    lines = ["| method | original | " + " | ".join(GENERATORS) + " |", "|---|---|" + "---|" * len(GENERATORS)]
    if benchmark == "re2rnn" and version == "clean":
        lines += [f"| {name} | {scores[dataset]:.2f} | " + " | ".join("–" for _ in GENERATORS) + " |" for name, scores in PUBLISHED.items()]
    done = expected = 0
    for method in grid["methods"]:
        if version not in ONLY_VERSIONS.get((benchmark, method), grid["versions"]):
            continue
        folder = root / "results" / benchmark / method
        original = score(folder / f"{dataset}.{version}.original.json", benchmark)
        cells = ["…" if original is None else f"{original:.{digits}f}"]
        done, expected = done + (original is not None), expected + 1
        for generator in GENERATORS:
            values = [v for v in (score(folder / f"{dataset}.{version}.{generator}{s}.json", benchmark) for s in seeds) if v is not None]
            cells.append(cell(original, values, len(seeds), digits))
            done, expected = done + len(values), expected + len(seeds)
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines, done, expected


def changes(root: Path, benchmark: str, dataset: str) -> list[str]:
    # per generator: rules/queries changed, how many of the replaced words sat inside an alternation, examples
    folder = root / benchmark / (dataset if benchmark == "re2rnn" else "")
    prefix = "rules" if benchmark == "re2rnn" else "queries"
    original = folder / f"{prefix}.original.tsv"
    if not original.exists():
        return []
    total = sum(1 for line in open(original) if line.strip())
    lines = ["| generator | changed (seed 1) | inside an alternation | examples |", "|---|---|---|---|"]
    for generator in GENERATORS:
        log = folder / f"{prefix}.{generator}1.log.tsv"
        if not log.exists():
            continue
        rows = [line.rstrip("\n").split("\t") for line in open(log) if line.strip()]
        inside = sum(r[3] == "1" for r in rows)
        examples = ", ".join(f"{r[1]}→{r[2]}" for r in rows[:6])
        lines.append(f"| {generator} | {len(rows)}/{total} | {inside}/{len(rows)} | {examples} |")
    return lines


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "output": "REPORT.md"})
    root = Path(p.data_root)
    body, done, expected = [], 0, 0
    for benchmark in ("re2rnn", "trec5"):
        title = "RE2RNN zero-shot rule classification — accuracy %" if benchmark == "re2rnn" else "TREC-5 Confusion track — MAP"
        body += [f"## {title}", ""]
        for dataset in GRID[benchmark]["datasets"]:
            body += [f"### {DATASET_NAMES[dataset]}", "", "Query variations:", ""] + changes(root, benchmark, dataset) + [""]
            for version in GRID[benchmark]["versions"]:
                lines, d, e = table(root, benchmark, dataset, version)
                done, expected = done + d, expected + e
                body += [f"**{VERSION_NAMES[version]}**", ""] + lines + [""]
    head = ["# Query augmentation: dre under query variations and document noise", "",
            f"Updated {time.strftime('%Y-%m-%d %H:%M')} — {done}/{expected} results.", "",
            "Query variations after Penha, Câmara & Hauff, *Evaluating the Robustness of Retrieval Pipelines with Query",
            "Variation Generators* (ECIR 2022): one word of every rule / query changed, each generator a separate ablation —",
            "`synonym` (WordNet, most frequent sense), `neighbour_swap`, `random_sub`, `qwerty_sub` (misspellings). RE2RNN: 3",
            "seeds per generator, cells are mean ± std (relative change to the original rules); TREC-5: 1 seed. Methods named",
            "`<method>.<engine>` are dre (matching.py); `phoc_csls100` is the spelling chain PHOC → GloVe 100d scored with",
            "CSLS. Baselines: published RE2RNN numbers (original rules, clean text only), `fuzzy` (`regex` with an edit",
            "budget), `re`, `bm25_word`, `bm25_char4` (TREC-5). Nothing is tuned on these benchmarks.", "",
            "Caveats: WordNet's most frequent sense is often not the rule's sense (e.g. fare→menu); a replaced word inside",
            "an alternation keeps its siblings, so the rule changes less (see \"inside an alternation\"); TREC-5 relevance",
            "is the exact occurrence of the original query string, so synonym queries are judged against the original",
            "word's documents and part of their drop is by construction, and many TREC-5 words (names, technical terms)",
            "have no synonym (see \"changed\").", ""]
    output = Path(p.output)
    part = output.with_name(f"{output.name}.{os.getpid()}.part")  # make -j: several reports at once
    part.write_text("\n".join(head + body) + "\n")
    part.replace(output)


if __name__ == "__main__":
    main()
