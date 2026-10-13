#!/usr/bin/env python3
# Query variations after Penha, Câmara & Hauff (ECIR 2022), one word per rule/query changed, every generator its own
# ablation:
#   synonym          (paraphrasing class) WordNet synonym: a single-word lemma of the word's most frequent sense that
#                    has one, neither the word nor an inflection of it
#   neighbour_swap   (misspelling class) two adjacent, different letters swapped
#   random_sub       (misspelling class) one letter replaced by a random other letter
#   qwerty_sub       (misspelling class) one letter replaced by a neighbouring key on a QWERTY keyboard
# Replaceable words are the rule's lower-case alphabetic tokens (never operators, ".", token classes, BOS/EOS); a
# replacement already present in the rule is rejected and the next candidate tried, so alternations never get
# duplicate alternatives. A rule without a replaceable word stays as it is.
# Inputs (read by path): RE2RNN rules of examples/semantics (label, RE2RNN rule, dre rule) and TREC-5 queries of
# examples/ir (qid, family, regex). Outputs, same layouts with the dre rule / regex only:
#   data/re2rnn/<dataset>/rules.<variant>.tsv   label <TAB> rule
#   data/trec5/queries.<variant>.tsv            qid <TAB> family <TAB> regex
# where variant is "original" or <generator><seed>, plus <variant>.log.tsv: id, word, replacement, whether the word
# was inside an alternation (1) or the only alternative at its position (0).
import random
import re
import string
from collections.abc import Callable
from pathlib import Path

import fargv
import nltk

WORD = re.compile(r"[a-z]+")
QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
GENERATORS = ("synonym", "neighbour_swap", "random_sub", "qwerty_sub")


def qwerty_neighbours() -> dict[str, str]:
    position = {c: (r, k) for r, row in enumerate(QWERTY_ROWS) for k, c in enumerate(row)}
    return {c: "".join(d for d, (s, l) in position.items() if d != c and abs(s - r) <= 1 and abs(l - k) <= 1)
            for c, (r, k) in position.items()}


QWERTY = qwerty_neighbours()


def synonyms(word: str, wordnet) -> list[str]:
    own = {word} | {form for pos in "nvar" for form in [wordnet.morphy(word, pos)] if form}
    for synset in wordnet.synsets(word):
        found = [lemma.lower() for lemma in synset.lemma_names()]
        found = [l for l in dict.fromkeys(found) if WORD.fullmatch(l) and l not in own and wordnet.morphy(l) not in own]
        if found:
            return found
    return []


def misspellings(kind: str, word: str, rng: random.Random) -> list[str]:
    # all candidates of one kind, shuffled
    if kind == "neighbour_swap":
        found = [word[:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len(word) - 1) if word[i] != word[i + 1]]
    elif kind == "random_sub":
        found = [word[:i] + rng.choice([c for c in string.ascii_lowercase if c != word[i]]) + word[i + 1:] for i in range(len(word))]
    else:
        found = [word[:i] + c + word[i + 1:] for i in range(len(word)) for c in QWERTY.get(word[i], "")]
    found = list(dict.fromkeys(found))
    rng.shuffle(found)
    return found


def in_alternation(tokens: list[str], position: int) -> bool:
    # whether the innermost group around tokens[position] has a "|" at its own depth
    depth, begin = 0, -1
    for n in range(position - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(tokens[n], 0)
        if depth < 0:
            begin = n
            break
    depth = 0
    for n in range(begin + 1, len(tokens)):
        if tokens[n] == "(":
            depth += 1
        elif tokens[n] == ")":
            if depth == 0:
                break
            depth -= 1
        elif tokens[n] == "|" and depth == 0:
            return True
    return False


def vary(rule: str, replacements: Callable[[str, random.Random], list[str]], rng: random.Random) -> tuple[str, str, str, int] | None:
    # (varied rule, word, replacement, inside alternation) or None when no word of the rule can be replaced
    tokens = rule.split()
    present = set(tokens)
    positions = [n for n, token in enumerate(tokens) if WORD.fullmatch(token)]
    rng.shuffle(positions)
    for position in positions:
        word = tokens[position]
        for replacement in replacements(word, rng):
            if replacement not in present:
                tokens[position] = replacement
                return " ".join(tokens), word, replacement, int(in_alternation(tokens, position))
    return None


def generator(name: str, wordnet) -> Callable[[str, random.Random], list[str]]:
    if name == "synonym":
        return lambda word, rng: rng.sample(found := synonyms(word, wordnet), len(found))
    return lambda word, rng: misspellings(name, word, rng) if len(word) >= 3 else []


def write_variants(rules: list[tuple[str, ...]], out_dir: Path, prefix: str, tag: str, generators: list[str], seeds: list[int], wordnet) -> None:
    # rules: tuples whose last element is the rule; the others are copied (label, or qid and family)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}.original.tsv").write_text("".join("\t".join(r) + "\n" for r in rules))
    for name in generators:
        replacements = generator(name, wordnet)
        for seed in seeds:
            rng = random.Random(f"{tag}/{name}/{seed}")
            out, log = [], []
            for record in rules:
                varied = vary(record[-1], replacements, rng)
                out.append("\t".join([*record[:-1], varied[0] if varied else record[-1]]) + "\n")
                if varied:
                    log.append(f"{record[0]}\t{varied[1]}\t{varied[2]}\t{varied[3]}\n")
            (out_dir / f"{prefix}.{name}{seed}.tsv").write_text("".join(out))
            (out_dir / f"{prefix}.{name}{seed}.log.tsv").write_text("".join(log))
            print(f"{tag} {name}{seed}: {len(log)}/{len(rules)} changed; e.g. "
                  + ", ".join("->".join(line.split("\t")[1:3]) for line in log[:5]), flush=True)


def main() -> None:
    p, _ = fargv.parse({"re2rnn_root": "../semantics/data", "trec_root": "../ir/data/trec5", "data_root": "data",
                        "datasets": "ATIS TREC SMS", "generators": " ".join(GENERATORS), "re2rnn_seeds": "1,2,3",
                        "trec_seeds": "1", "nltk_data": "data/nltk_data"})
    nltk.data.path.insert(0, str(Path(p.nltk_data).resolve()))
    from nltk.corpus import wordnet
    generators = p.generators.split()
    for dataset in p.datasets.split():
        lines = [l.rstrip("\n").split("\t") for l in open(Path(p.re2rnn_root) / dataset / "rules.tsv") if l.strip()]
        write_variants([(label, rule) for label, _, rule in lines], Path(p.data_root) / "re2rnn" / dataset, "rules",
                       f"re2rnn/{dataset}", generators, [int(s) for s in p.re2rnn_seeds.split(",")], wordnet)
    queries = [tuple(l.rstrip("\n").split("\t", 2)) for l in open(Path(p.trec_root) / "queries.tsv") if l.strip()]
    write_variants(queries, Path(p.data_root) / "trec5", "queries", "trec5", generators,
                   [int(s) for s in p.trec_seeds.split(",")], wordnet)


if __name__ == "__main__":
    main()
