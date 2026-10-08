# Searching a collection of texts with word regexes: the texts are tokenized once into a Corpus, one observer over the
# corpus vocabulary serves every query (its cache keeps the distances to every query word seen), and a matcher scores
# documents or finds the matches, reported as character spans in the texts.
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from .abstract_matcher import AbstractMatcher, ViterbiMatcher
from .abstract_observer import SequenceObserver
from .fa_graph import Automaton
from .tokenizing import Corpus
from .wordre_parser import compile_regex


@dataclass
class Hit:
    document: int
    start: int  # character offsets in the document's text
    end: int
    probability: float


class Searcher(object):
    def __init__(self, corpus: Corpus, observer: SequenceObserver, matcher: Optional[AbstractMatcher] = None) -> None:
        if observer.vocabulary is not corpus.vocabulary:
            raise ValueError("The observer must observe the corpus vocabulary: build it with corpus.vocabulary")
        self.corpus = corpus
        self.observer = observer
        self.matcher = ViterbiMatcher() if matcher is None else matcher

    def scores(self, regex: Union[str, Automaton]) -> torch.Tensor:
        # (D,): each document's best match
        automaton = compile_regex(regex) if isinstance(regex, str) else regex
        return self.matcher.scores(automaton, self.observer.observe(automaton), self.corpus.documents)

    def search(self, regex: Union[str, Automaton], top_n: Optional[int] = None, min_probability: float = 0.01) -> List[Hit]:
        # Matches over the whole collection, most probable first. Overlapping matches are resolved greedily: the most
        # probable first, then the earliest, then the longest; a match overlapping a kept one is dropped.
        automaton = compile_regex(regex) if isinstance(regex, str) else regex
        documents, starts, ends, probabilities = self.matcher.matches(automaton, self.observer.observe(automaton), self.corpus.documents, min_probability)
        candidates = sorted(zip(documents.tolist(), starts.tolist(), ends.tolist(), probabilities.tolist()),
                            key=lambda m: (-m[3], m[0], m[1], m[1] - m[2]))
        kept = {}  # document -> kept token spans
        hits = []
        for document, start, end, probability in candidates:
            spans = kept.setdefault(document, [])
            if any(start < other_end and other_start < end for other_start, other_end in spans):
                continue
            spans.append((start, end))
            hits.append(Hit(document, *self.corpus.char_span(document, start, end), probability))
            if top_n is not None and len(hits) == top_n:
                break
        return hits
