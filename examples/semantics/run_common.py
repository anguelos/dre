# Shared by the methods: classify one dataset version with RE2RNN's rules, scoring every rule on every sentence with
# dre (whole-sentence matching, as their RE system does) under the given observer, then aggregating rule scores
# into labels with their priorities. Writes one line per sentence: gold label, predicted label.
import json
from pathlib import Path
from collections.abc import Callable

import torch

from dre import Corpus, DFAMatcher, SequenceObserver, Tokenizer, ViterbiMatcher, Vocabulary, compile_regex, query_words
from re2rnn_rules import aggregate

# Given the sentence vocabulary, returns the observer of a method.
ObserverFactory = Callable[[Vocabulary], SequenceObserver]
MATCHERS = {"viterbi": ViterbiMatcher, "dfa": DFAMatcher}


def read_sentences(path: Path) -> tuple[list[str], list[list[str]]]:
    records = [json.loads(line) for line in open(path)]
    return [r["label"] for r in records], [r["tokens"] for r in records]


def read_dre_rules(path: Path) -> list[tuple[str, str]]:
    rules = []
    for line in open(path):
        label, _, dre_rule = line.rstrip("\n").split("\t")
        rules.append((label, dre_rule))
    return rules


def classify(data_root: str, dataset: str, version: str, output: str, make_observer: ObserverFactory, device: str, engine: str = "viterbi",
             rules: str = "rules.tsv") -> None:
    # rules: the rule file inside the dataset directory (rules.tsv or a query variation of it)
    root = Path(data_root) / dataset
    labels = json.loads((root / "labels.json").read_text())
    split = "dev" if version == "dev" else "test"
    gold, sentences = read_sentences(root / f"{split}.{'clean' if version == 'dev' else version}.jsonl")
    rules = read_dre_rules(root / rules)
    automata = [compile_regex(rule) for _, rule in rules]

    corpus = Corpus([" ".join(sentence) for sentence in sentences], make_observer, MATCHERS[engine](anchored=True, device=torch.device(device)),
                    Tokenizer.pretokenized())
    corpus.observer.prepare(sorted({word for automaton in automata for word in query_words(automaton)}))

    scores = torch.stack([corpus.scores(automaton) for automaton in automata], dim=1)
    predictions = aggregate(scores.tolist(), [label for label, _ in rules], labels, dataset)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for truth, predicted in zip(gold, predictions):
            f.write(f"{truth}\t{labels[predicted]}\n")
