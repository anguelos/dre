# Non-dre baselines on the same rules/queries:
#   re          Python `re` on the token string (every token followed by one space): match count per document
#   fuzzy       the `regex` package with an edit budget k = max(1, floor(0.2 * letters of the required words)), edits
#               counted on the token string; RE2RNN: the rule must cover the sentence (score 1), TREC-5: minus the
#               edits of the best match. A match taking longer than `timeout` seconds counts as no match.
#   bm25_word   BM25 (k1 1.2, b 0.75, idf ln(1 + (N - df + .5) / (df + .5))) over tokens; a query is the bag of its
#   bm25_char4  words / of the character 4-grams of its words marked "_word_" (McNamee & Mayfield). Order and
#               operators are ignored, which is what makes BM25 a baseline here. Only the query terms are counted
#               (one pass over the documents), so no index is built.
import math
import multiprocessing
import re
from collections import Counter
from collections.abc import Callable, Iterable

from benchmarks import PUNCTUATIONS

OPERATORS = set("()|*?+")


def python_regex(rule: str) -> str:
    # body of a dre word regex as a Python regex over a token string; every atom keeps its trailing space
    parts = []
    for token in rule.split():
        if token in OPERATORS:
            parts.append("(?:" if token == "(" else token)
        elif token == ".":
            parts.append(r"(?:[^ ]+ )")
        elif token == "<punct>":
            parts.append("(?:(?:" + "|".join(re.escape(p) for p in sorted(PUNCTUATIONS)) + ") )")
        elif token == "<number>":
            parts.append(r"(?:(?:\d+\.?\d*|\.\d+) )")
        else:
            parts.append("(?:" + re.escape(token.lower()) + " )")
    return "".join(parts)


def required_words(rule: str) -> list[str]:
    # top-level words not made optional by a following ? or *; a top-level | makes nothing required
    tokens, depth, required = rule.split(), 0, []
    for n, token in enumerate(tokens):
        depth += {"(": 1, ")": -1}.get(token, 0)
        if token == "|" and depth == 0:
            return []
        if depth == 0 and token.isalpha() and token.islower() and (n + 1 == len(tokens) or tokens[n + 1] not in "?*"):
            required.append(token)
    return required


def edit_budget(rule: str, rate: float = 0.2) -> int:
    words = required_words(rule) or [t for t in rule.split() if t.isalpha() and t.islower()]
    return max(1, math.floor(rate * sum(len(w) for w in words)))


def token_string(tokens: Iterable[str]) -> str:
    return "".join(token.lower() + " " for token in tokens)


def parallel(function: Callable, items: Iterable, workers: int, chunksize: int = 16):
    if workers <= 1:
        yield from map(function, items)
        return
    with multiprocessing.Pool(workers) as pool:
        yield from pool.imap(function, items, chunksize=chunksize)


# ---- RE2RNN: one score per (sentence, rule)

def fuzzy_rule_scores(sentences: list[list[str]], rules: list[str], timeout: float = 2.0) -> tuple[list[list[float]], int]:
    import regex
    patterns = [regex.compile("(?:" + python_regex(rule) + "){e<=%d}" % edit_budget(rule), flags=regex.BESTMATCH) for rule in rules]
    scores, timeouts = [], 0
    for sentence in sentences:
        text, row = token_string(sentence), []
        for pattern in patterns:
            try:
                row.append(float(pattern.fullmatch(text, timeout=timeout) is not None))
            except TimeoutError:
                row.append(0.0)
                timeouts += 1
        scores.append(row)
    return scores, timeouts


# ---- TREC-5: per query, docno -> score (only documents with a match)

class _ReScorer(object):
    def __init__(self, queries: list[tuple[str, str]]) -> None:
        self.queries = queries
        self.patterns = None

    def __call__(self, doc: tuple[str, str]) -> tuple[str, list[tuple[str, float]]]:
        if self.patterns is None:
            self.patterns = [(qid, re.compile(r"(?<![^ ])(?:" + python_regex(regex) + ")"), required_words(regex)) for qid, regex in self.queries]
        text = doc[1] + " "
        padded = " " + text
        found = []
        for qid, pattern, required in self.patterns:
            if all(f" {w} " in padded for w in required):
                count = sum(1 for _ in pattern.finditer(text))
                if count:
                    found.append((qid, float(count)))
        return doc[0], found


class _FuzzyScorer(_ReScorer):
    def __call__(self, doc: tuple[str, str]) -> tuple[str, list[tuple[str, float]]]:
        import regex
        if self.patterns is None:
            self.patterns = [(qid, regex.compile(r"(?<![^ ])(?:" + python_regex(rx) + r"){e<=%d}" % edit_budget(rx), flags=regex.BESTMATCH))
                             for qid, rx in self.queries]
        found = []
        for qid, pattern in self.patterns:
            match = pattern.search(doc[1] + " ")
            if match is not None:
                found.append((qid, -float(sum(match.fuzzy_counts))))
        return doc[0], found


def regex_document_scores(kind: str, docs: Iterable[tuple[str, str]], queries: list[tuple[str, str]], workers: int) -> dict[str, dict[str, float]]:
    # queries: (key, regex); kind: re or fuzzy
    scorer = _ReScorer(queries) if kind == "re" else _FuzzyScorer(queries)
    scores: dict[str, dict[str, float]] = {key: {} for key, _ in queries}
    for docno, found in parallel(scorer, docs, workers):
        for key, score in found:
            scores[key][docno] = score
    return scores


def word_terms(words: list[str]) -> list[str]:
    return words


def char4_terms(words: list[str]) -> list[str]:
    marked = ["_" + w + "_" for w in words]
    return [m[i:i + 4] for m in marked for i in range(max(1, len(m) - 3))]


TERMS = {"bm25_word": word_terms, "bm25_char4": char4_terms}


class _TermCounter(object):
    def __init__(self, terms: Callable[[list[str]], list[str]], vocabulary: set[str]) -> None:
        self.terms, self.vocabulary = terms, vocabulary

    def __call__(self, doc: tuple[str, str]) -> tuple[str, int, dict[str, int]]:
        terms = self.terms(doc[1].split())
        return doc[0], len(terms), dict(Counter(t for t in terms if t in self.vocabulary))


def bm25_document_scores(kind: str, docs: Iterable[tuple[str, str]], queries: list[tuple[str, str]], workers: int,
                         k1: float = 1.2, b: float = 0.75) -> dict[str, dict[str, float]]:
    terms = TERMS[kind]
    bags = {key: Counter(terms([t for t in regex.split() if t.isalpha() and t.islower()])) for key, regex in queries}
    vocabulary = {t for bag in bags.values() for t in bag}
    lengths, postings = {}, {t: {} for t in vocabulary}
    for docno, length, counts in parallel(_TermCounter(terms, vocabulary), docs, workers, chunksize=64):
        lengths[docno] = length
        for t, c in counts.items():
            postings[t][docno] = c
    n, average = len(lengths), sum(lengths.values()) / max(1, len(lengths))
    scores: dict[str, dict[str, float]] = {}
    for key, bag in bags.items():
        total: dict[str, float] = {}
        for t, qtf in bag.items():
            idf = math.log(1 + (n - len(postings[t]) + 0.5) / (len(postings[t]) + 0.5))
            for docno, tf in postings[t].items():
                total[docno] = total.get(docno, 0.0) + qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[docno] / average))
        scores[key] = total
    return scores
