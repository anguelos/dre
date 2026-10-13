# The two benchmarks, read by path from the other examples' data:
#   re2rnn  RE2RNN zero-shot rule classification (Jiang et al., EMNLP 2020; examples/semantics/data): a sentence
#           matches a rule only if the rule covers it entirely; a label scores the summed scores of its rules minus,
#           for every (dominant, dominated) priority, the dominant label's total; arg-max, ties to the lowest index.
#   trec5   TREC-5 Confusion track documents (examples/ir/data/trec5): rank documents by the best match of the query
#           regex; relevance = the query string occurs in the original text. AP, nDCG@10, P@10, R@1000, RR.
# Every run handles all query variants of one (method, dataset, document version) at once: the corpus is loaded and
# the observer prepared once, for the words of all variants.
import json
import math
from collections.abc import Iterator
from pathlib import Path

import torch

import dre
from matching import ENGINES

# RE2RNN's punctuation set and number test (their fsa_to_tensor.py) and label priorities (create_logic_mat_bias.py)
PUNCTUATIONS = {",", "，", ":", "：", "!", "！", "《", "》", "。", "；", ".", "(", ")", "（", "）", "|", "?", '"'}
CLASSES = {"<number>": lambda token: token.replace(".", "", 1).isdigit(), "<punct>": lambda token: token in PUNCTUATIONS}
PRIORITIES = {
    "ATIS": [("ground_fare", "ground_service"), ("flight", "city"), ("flight", "airport"), ("flight", "airline"), ("flight", "meal"),
             ("flight", "capacity"), ("flight", "cheapest"), ("flight", "distance"), ("airfare", "flight"), ("flight", "flight+airline"),
             ("abbreviation", "restriction"), ("ground_service", "airport")],
    "TREC": [("ENTITY", "DESCRIPTION")],
    "SMS": [("spam", "ham")],
}
TREC5_FAMILIES = ("word", "phrase", "pattern")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=1))
    part.replace(path)


# ---- RE2RNN

def read_sentences(re2rnn_root: str, dataset: str, version: str) -> tuple[list[str], list[str], list[list[str]]]:
    # (labels, gold label per sentence, tokens per sentence) of the test set version (clean, noise5, noise20)
    root = Path(re2rnn_root) / dataset
    records = [json.loads(line) for line in open(root / f"test.{version}.jsonl")]
    return json.loads((root / "labels.json").read_text()), [r["label"] for r in records], [r["tokens"] for r in records]


def read_rules(path: Path) -> list[tuple[str, str]]:
    return [tuple(line.rstrip("\n").split("\t")) for line in open(path) if line.strip()]


def aggregate(scores: list[list[float]], rule_labels: list[str], labels: list[str], dataset: str) -> list[int]:
    index = {label: n for n, label in enumerate(labels)}
    predictions = []
    for row in scores:
        totals = [0.0] * len(labels)
        for score, label in zip(row, rule_labels):
            totals[index[label]] += score
        adjusted = list(totals)
        for dominant, dominated in PRIORITIES[dataset]:
            adjusted[index[dominated]] -= totals[index[dominant]]
        predictions.append(max(range(len(labels)), key=lambda n: (adjusted[n], -n)))
    return predictions


def re2rnn_result(rule_scores: list[list[float]], rules: list[tuple[str, str]], labels: list[str], gold: list[str],
                  dataset: str) -> tuple[dict, list[str]]:
    predicted = [labels[n] for n in aggregate(rule_scores, [label for label, _ in rules], labels, dataset)]
    return {"n": len(gold), "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold)}, predicted


# ---- TREC-5

def read_tokens(trec_root: str, version: str) -> Iterator[tuple[str, str]]:
    # (docno, lower-cased tokens joined by spaces)
    with open(Path(trec_root) / "tokens" / f"{version}.tsv", encoding="utf-8", errors="replace") as f:
        for line in f:
            docno, tokens = line.rstrip("\n").split("\t", 1)
            yield docno, tokens


def read_queries(path: Path) -> list[tuple[str, str, str]]:
    return [tuple(line.rstrip("\n").split("\t", 2)) for line in open(path) if line.strip()]


def read_qrels(trec_root: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for line in open(Path(trec_root) / "qrels.txt"):
        qid, _, docno, relevance = line.split()
        if int(relevance) > 0:
            qrels.setdefault(qid, set()).add(docno)
    return qrels


def rank(scores: dict[str, float], depth: int = 1000) -> list[str]:
    # by score, ties by docno (as examples/ir writes its run files)
    return [docno for docno, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:depth]]


def measures(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    hits = [docno in relevant for docno in ranked]
    precisions = [sum(hits[:k + 1]) / (k + 1) for k, hit in enumerate(hits) if hit]
    ideal = sum(1 / math.log2(k + 2) for k in range(min(10, len(relevant))))
    return {"AP": sum(precisions) / len(relevant),
            "nDCG@10": sum(1 / math.log2(k + 2) for k, hit in enumerate(hits[:10]) if hit) / ideal,
            "P@10": sum(hits[:10]) / 10,
            "R@1000": sum(hits) / len(relevant),
            "RR": next((1 / (k + 1) for k, hit in enumerate(hits) if hit), 0.0)}


def trec5_result(run: dict[str, list[str]], queries: list[tuple[str, str, str]], qrels: dict[str, set[str]]) -> dict:
    # means per family and over all queries, plus the per-query values
    per_query = {qid: measures(run.get(qid, []), qrels[qid]) for qid, _, _ in queries if qid in qrels}
    families = {qid: family for qid, family, _ in queries}
    groups = {f: [q for q in per_query if families[q] == f] for f in TREC5_FAMILIES} | {"all": list(per_query)}
    means = {group: {m: sum(per_query[q][m] for q in qids) / len(qids) for m in next(iter(per_query.values()))}
             for group, qids in groups.items() if qids}
    return {"n": len(per_query), "means": means, "per_query": per_query}


def write_run(path: Path, run: dict[str, list[str]], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid in sorted(run):
            for position, docno in enumerate(run[qid], start=1):
                f.write(f"{qid} Q0 {docno} {position} {-position} {tag}\n")


# ---- dre

def dre_corpus(texts: list[str], make_observer, engine: str, anchored: bool, device: str) -> dre.Corpus:
    return dre.Corpus(texts, make_observer, ENGINES[engine](anchored=anchored, device=torch.device(device)), dre.Tokenizer.pretokenized())


def dre_prepare(corpus: dre.Corpus, rule_sets: list[list[str]]) -> list[list[dre.Automaton]]:
    automata = [[dre.compile_regex(rule) for rule in rules] for rules in rule_sets]
    corpus.observer.prepare(sorted({w for group in automata for a in group for w in dre.query_words(a)}))
    return automata
