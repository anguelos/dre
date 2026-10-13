import re
import string
import warnings
from collections.abc import Iterable, Iterator

import numpy as np
import torch
from pylelemmatize.fast_mapper import LemmatizerBMP
from torch import Tensor

from .optional import optional_import, warn_missing

nltk = optional_import("nltk")


def naive_sentence_tokenize(text: str) -> list[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def naive_word_tokenize(sentence: str) -> list[str]:
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", sentence)


def words_of_sentences(db: str, tokenizer: str = "auto") -> Iterator[list[str]]:
    if tokenizer == "auto":
        tokenizer = "naive"
        if not nltk:
            warn_missing("nltk", "the built-in naive tokenizer")
        else:
            try:
                nltk.word_tokenize("probe.")
                tokenizer = "nltk"
            except LookupError:  # nltk installed without its punkt data
                warnings.warn("nltk is installed without its punkt data: python -m nltk.downloader punkt_tab. "
                              "Falling back to the built-in naive tokenizer.", stacklevel=2)
    if tokenizer == "nltk":
        sent_tokenize, word_tokenize = nltk.sent_tokenize, nltk.word_tokenize
    elif tokenizer == "naive":
        sent_tokenize, word_tokenize = naive_sentence_tokenize, naive_word_tokenize
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer!r}, expected 'auto', 'nltk' or 'naive'")
    for sentence in sent_tokenize(db):
        yield word_tokenize(sentence)


END_OF_DOCUMENT = "\x1e"  # ASCII record separator: joins the documents of a corpus; always token id 0


class Vocabulary(object):
    # Strings <-> token ids. Id 0 is always END_OF_DOCUMENT, on which a matcher returns to the automaton's initial
    # state. The front end encodes texts with it; the engine (observers' caches, matchers) only sees ids.
    def __init__(self, tokens: Iterable[str]) -> None:
        self.tokens = [END_OF_DOCUMENT] + [token for token in dict.fromkeys(tokens) if token != END_OF_DOCUMENT]
        self.index = {token: n for n, token in enumerate(self.tokens)}

    @classmethod
    def from_documents(cls, documents: Iterable[list[str]]) -> "Vocabulary":
        return cls(sorted({token for document in documents for token in document}))

    @classmethod
    def from_spans(cls, text: str, spans: Tensor, chunk: int = 1 << 20) -> tuple["Vocabulary", Tensor]:
        # The distinct tokens text[start:end] of spans (2, T), sorted, and each token's id (T,). Tokens are identified
        # by a hash of their unicode code points computed with tensor operations, so only one string per distinct token
        # is ever made; every token is then compared with its type's first occurrence, so the result is exact.
        code_points = torch.from_numpy(np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int32))  # < 2^21
        starts, lengths = spans[0], spans[1] - spans[0]
        hashes = torch.cat([_span_hashes(code_points, starts[n:n + chunk], lengths[n:n + chunk]) for n in range(0, starts.numel(), chunk)]
                           or [torch.zeros(0, dtype=torch.long)])
        types, type_ids = torch.unique(hashes, return_inverse=True)
        first = torch.full((types.numel(),), starts.numel(), dtype=torch.long).scatter_reduce_(0, type_ids, torch.arange(starts.numel()), reduce="amin")
        for n in range(0, starts.numel(), chunk):  # exactness: every token equals its type's first occurrence
            if not _same_spans(code_points, starts[n:n + chunk], lengths[n:n + chunk], starts[first[type_ids[n:n + chunk]]], lengths[first[type_ids[n:n + chunk]]]):
                raise RuntimeError("Token hash collision; please report it")
        strings = [text[start:start + length] for start, length in zip(starts[first].tolist(), lengths[first].tolist())]
        vocabulary = cls(sorted(strings))
        rank = torch.tensor([vocabulary.index[s] for s in strings], dtype=torch.long)
        return vocabulary, rank[type_ids]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, n: int) -> str:
        return self.tokens[n]

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def encode(self, tokens: list[str]) -> Tensor:
        return torch.tensor([self.index[token] for token in tokens], dtype=torch.long)


_HASH_PRIME = (1 << 31) - 1
_HASH_BASES = (131, 137)


def _span_hashes(code_points: Tensor, starts: Tensor, lengths: Tensor) -> Tensor:
    # Two polynomial hashes modulo 2^31 - 1 of each span's code points, combined with its length into one int64.
    token = torch.repeat_interleave(torch.arange(starts.numel()), lengths)                          # (C,) token of each character
    offset = torch.arange(token.numel()) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    characters = code_points[starts[token] + offset].long()
    combined = lengths.clone()
    for base in _HASH_BASES:
        powers = torch.ones(int(lengths.max()) + 1 if lengths.numel() else 1, dtype=torch.long)
        for n in range(1, powers.numel()):
            powers[n] = powers[n - 1] * base % _HASH_PRIME
        terms = characters * powers[offset] % _HASH_PRIME
        hashes = torch.zeros(starts.numel(), dtype=torch.long).index_add_(0, token, terms) % _HASH_PRIME
        combined = combined * _HASH_PRIME + hashes  # int64 may wrap here: that only risks collisions, which are checked
    return combined


def _same_spans(code_points: Tensor, starts: Tensor, lengths: Tensor, other_starts: Tensor, other_lengths: Tensor) -> bool:
    if not torch.equal(lengths, other_lengths):
        return False
    token = torch.repeat_interleave(torch.arange(starts.numel()), lengths)
    offset = torch.arange(token.numel()) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    return torch.equal(code_points[starts[token] + offset], code_points[other_starts[token] + offset])


# Character categories of the Tokenizer, one per code of its mapper's alphabet.
SIMPLE = 0     # part of a token (letters, digits, and the unknown code: characters outside the alphabet)
KEPT = 1       # separates tokens and is a token itself (punctuation)
DUMPED = 2     # separates tokens and is dropped (whitespace, quotes)
ATTACHING = 3  # starts a token with the simple characters that follow it ("what's" -> what | 's); alone, it is KEPT
END = 4        # END_OF_DOCUMENT: always a token of its own


class Tokenizer(object):
    # Splits text into word-like atoms: returns their (start, end) character offsets, nothing else (no normalization:
    # a token is text[start:end]). The text is encoded with a pylelemmatize mapper (one code per character, so code
    # positions are text positions) and every code has a category; tensor operations find the boundaries.
    # merge_kept: consecutive kept separators form one token ("?!", "...").
    def __init__(self, mapper: LemmatizerBMP | None = None, categories: Tensor | None = None, merge_kept: bool = True) -> None:
        self.mapper = self.default_mapper() if mapper is None else mapper
        self.categories = self.default_categories(self.mapper) if categories is None else categories
        self.merge_kept = merge_kept

    @classmethod
    def pretokenized(cls, mapper: LemmatizerBMP | None = None) -> "Tokenizer":
        # For texts already tokenized into whitespace-separated tokens: whitespace separates, everything else is kept.
        mapper = cls.default_mapper() if mapper is None else mapper
        def category(c: str) -> int:
            return END if c == END_OF_DOCUMENT else DUMPED if c.isspace() else SIMPLE
        return cls(mapper, torch.tensor([SIMPLE] + [category(c) for c in mapper.dst_alphabet_str], dtype=torch.uint8))

    @staticmethod
    def default_mapper() -> LemmatizerBMP:
        return LemmatizerBMP(string.ascii_letters + string.digits + string.punctuation + string.whitespace + END_OF_DOCUMENT
                             + "\u2018\u2019\u201c\u201d\u201e\u00ab\u00bb\u2013\u2014\u2026")

    @staticmethod
    def default_categories(mapper: LemmatizerBMP) -> Tensor:
        # code 0 is the unknown character; code k is the k-th character of the mapper's alphabet
        def category(c: str) -> int:
            if c == END_OF_DOCUMENT:
                return END
            if c.isspace() or c in "\"\u201c\u201d\u201e\u00ab\u00bb\u2018":
                return DUMPED
            if c in "'\u2019":
                return ATTACHING
            return SIMPLE if c.isalnum() else KEPT
        return torch.tensor([SIMPLE] + [category(c) for c in mapper.dst_alphabet_str], dtype=torch.uint8)

    def encode(self, text: str) -> Tensor:
        codes = torch.from_numpy(self.mapper.str_to_intlabel_seq(text).astype(np.int32))
        if codes.numel() != len(text):
            raise ValueError("The mapper changed the text's length (unicode normalization?); spans would not match the text")
        return codes

    def tokenize(self, text: str, device="cpu") -> Tensor:
        # Token spans (2, T): start and end (exclusive) character offsets in text.
        codes = self.encode(text).to(device)
        n = codes.numel()
        if n == 0:
            return torch.zeros(2, 0, dtype=torch.long)
        category = self.categories.to(device)[codes]                                                # (N,) uint8
        del codes
        followed_by_simple = torch.zeros(n, dtype=torch.bool, device=device)
        followed_by_simple[:-1] = category[1:] == SIMPLE
        category[(category == ATTACHING) & ~followed_by_simple] = KEPT
        del followed_by_simple
        previous = torch.cat([category.new_full((1,), DUMPED), category[:-1]])
        in_token = category != DUMPED
        starts = in_token & ((previous == DUMPED) | (previous == END) | (category == ATTACHING) | (category == END)
                             | ((category == SIMPLE) & (previous == KEPT))
                             | ((category == KEPT) & ((previous != KEPT) | (not self.merge_kept))))
        del previous, category
        ends = in_token.clone()  # last character of a token: the next one is outside any token or starts a new one
        ends[:-1] &= ~in_token[1:] | starts[1:]
        return torch.stack([starts.nonzero().reshape(-1), ends.nonzero().reshape(-1) + 1]).cpu()
