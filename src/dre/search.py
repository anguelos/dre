# Searching a collection of texts with word regexes. The Corpus keeps the collection as one string, the documents
# joined by END_OF_DOCUMENT, tokenizes it once and identifies its tokens; the engine only ever sees that one token
# sequence. The Corpus maps what the engine returns per token back to documents and character spans: it filters and
# ranks matches (search) and scores documents (retrieval).
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .abstract_matcher import AbstractMatcher, Matches, ViterbiMatcher
from .abstract_observer import ExactObserver, SequenceObserver
from .fa_graph import Automaton
from .tokenizing import END_OF_DOCUMENT, Tokenizer, Vocabulary
from .wordre_parser import compile_regex


@dataclass
class Hit:
    document: int
    start: int  # character offsets in the document's text
    end: int
    probability: float
    text: str


class Corpus(object):
    # make_observer: builds the observer of the corpus vocabulary (an observer class such as ExactObserver, or e.g.
    # lambda vocabulary: JointSoftmaxObserver(vocabulary, PHOCEmbedder(), L1Distance())); its cache serves every query.
    def __init__(self, texts: Iterable[str], make_observer: Callable[[Vocabulary], SequenceObserver] = ExactObserver,
                 matcher: AbstractMatcher | None = None, tokenizer: Tokenizer | None = None, device="cpu") -> None:
        texts = [text.replace(END_OF_DOCUMENT, " ") for text in texts]  # the separator cannot occur inside a document
        self.text = "".join(text + END_OF_DOCUMENT for text in texts)
        self.char_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(torch.tensor([len(t) + 1 for t in texts], dtype=torch.long), 0)])
        self.spans = (Tokenizer() if tokenizer is None else tokenizer).tokenize(self.text, device)    # (2, T)
        self.vocabulary, self.token_ids = Vocabulary.from_spans(self.text, self.spans)
        ends = (self.token_ids == 0).nonzero().reshape(-1)                                             # one per document
        self.token_offsets = torch.cat([torch.zeros(1, dtype=torch.long), ends + 1])                  # (D+1,)
        self.observer = make_observer(self.vocabulary)
        self.matcher = ViterbiMatcher() if matcher is None else matcher

    def __len__(self) -> int:
        return self.token_offsets.numel() - 1

    def document(self, n: int) -> str:
        return self.text[self.char_offsets[n]:self.char_offsets[n + 1] - 1]

    def matches(self, regex: str | Automaton, min_probability: float | None = 0.01) -> Matches:
        # The engine's answer for the whole collection: per token, the probability that a match ends there.
        automaton = compile_regex(regex, self.observer.normalize) if isinstance(regex, str) else regex
        return self.matcher.run(automaton, self.observer.observe(automaton), self.token_ids, min_probability)

    def scores(self, regex: str | Automaton) -> torch.Tensor:
        # (D,): each document's best match (anchored matchers: the probability that the whole document matches).
        end_probability = self.matches(regex, min_probability=None).end_probability
        if self.matcher.anchored:  # the last token of each document, if it has any
            last = self.token_offsets[1:] - 2
            return torch.where(last >= self.token_offsets[:-1], end_probability[last.clamp(min=0)], 0.0)
        document = torch.repeat_interleave(torch.arange(len(self)), torch.diff(self.token_offsets))
        return torch.zeros(len(self)).scatter_reduce_(0, document, end_probability, reduce="amax")

    def search(self, regex: str | Automaton, min_probability: float = 0.5, order: str = "probability") -> list[Hit]:
        # Matches with at least min_probability, most probable first (order="position": by document and position).
        # Consecutive tokens often end the same match ("a b +" on "a b b b"), so a match is credited to the local
        # maxima of the end probability (on a plateau, its last token: the longest match); then matches overlapping a
        # more probable one are dropped.
        matches = self.matches(regex, min_probability)
        p = matches.end_probability
        before = torch.cat([torch.zeros(1), p[:-1]])
        after = torch.cat([p[1:], torch.zeros(1)])
        ends = ((p >= min_probability) & (p > 0) & (p >= before) & (p > after) & (matches.start >= 0)).nonzero().reshape(-1)
        candidates = sorted(zip(p[ends].tolist(), matches.start[ends].tolist(), ends.tolist()), key=lambda m: (-m[0], m[1], m[1] - m[2]))
        kept: list[tuple[int, int]] = []
        hits = []
        for probability, start, end in candidates:
            if any(start <= other_end and other_start <= end for other_start, other_end in kept):
                continue
            kept.append((start, end))
            document = int(torch.searchsorted(self.token_offsets, start, right=True)) - 1
            offset = int(self.char_offsets[document])
            first, last = int(self.spans[0, start]), int(self.spans[1, end])
            hits.append(Hit(document, first - offset, last - offset, probability, self.text[first:last]))
        if order == "position":
            hits.sort(key=lambda hit: (hit.document, hit.start))
        elif order != "probability":
            raise ValueError(f"order must be 'probability' or 'position', got {order!r}")
        return hits
