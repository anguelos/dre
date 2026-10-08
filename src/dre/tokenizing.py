import re
import string
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from pylelemmatize.fast_mapper import LemmatizerBMP
from torch import Tensor

try:
    import nltk
except ImportError:
    nltk = None


def naive_sentence_tokenize(text: str) -> List[str]:
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def naive_word_tokenize(sentence: str) -> List[str]:
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", sentence)


def words_of_sentences(db: str, tokenizer: str = "auto") -> Iterator[List[str]]:
    if tokenizer == "auto":
        tokenizer = "naive"
        if nltk is not None:
            try:
                nltk.word_tokenize("probe.")
                tokenizer = "nltk"
            except LookupError:  # nltk installed without its punkt data
                pass
    if tokenizer == "nltk":
        if nltk is None:
            raise ImportError("tokenizer='nltk' requires nltk: pip install dre[nltk]")
        sent_tokenize, word_tokenize = nltk.sent_tokenize, nltk.word_tokenize
    elif tokenizer == "naive":
        sent_tokenize, word_tokenize = naive_sentence_tokenize, naive_word_tokenize
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer!r}, expected 'auto', 'nltk' or 'naive'")
    for sentence in sent_tokenize(db):
        yield word_tokenize(sentence)


class Vocabulary(object):
    # Strings <-> token ids. The front end encodes documents with it; the engine (observers' caches, matchers) only
    # sees ids.
    def __init__(self, tokens: Iterable[str]) -> None:
        self.tokens = list(dict.fromkeys(tokens))
        self.index = {token: n for n, token in enumerate(self.tokens)}

    @classmethod
    def from_documents(cls, documents: Iterable[List[str]]) -> "Vocabulary":
        return cls(sorted({token for document in documents for token in document}))

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, n: int) -> str:
        return self.tokens[n]

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)

    def encode(self, tokens: List[str]) -> Tensor:
        return torch.tensor([self.index[token] for token in tokens], dtype=torch.long)


# Character categories of the Tokenizer, one per code of its mapper's alphabet.
SIMPLE = 0     # part of a token (letters, digits, and the unknown code: characters outside the alphabet)
KEPT = 1       # separates tokens and is a token itself (punctuation)
DUMPED = 2     # separates tokens and is dropped (whitespace, quotes)
ATTACHING = 3  # starts a token with the simple characters that follow it ("what's" -> what | 's); alone, it is KEPT


class Tokenizer(object):
    # Splits texts into token spans with tensor operations: a text is encoded with a pylelemmatize mapper (one code per
    # character, so positions in the codes are positions in the text) and every code has a category.
    # merge_kept: consecutive kept separators form one token ("?!", "...").
    def __init__(self, mapper: Optional[LemmatizerBMP] = None, categories: Optional[Tensor] = None, merge_kept: bool = True) -> None:
        self.mapper = self.default_mapper() if mapper is None else mapper
        self.categories = self.default_categories(self.mapper) if categories is None else categories
        self.merge_kept = merge_kept

    @classmethod
    def pretokenized(cls, mapper: Optional[LemmatizerBMP] = None) -> "Tokenizer":
        # For texts already tokenized into whitespace-separated tokens: whitespace separates, everything else is kept.
        mapper = cls.default_mapper() if mapper is None else mapper
        categories = torch.tensor([SIMPLE] + [DUMPED if c.isspace() else SIMPLE for c in mapper.dst_alphabet_str], dtype=torch.uint8)
        return cls(mapper, categories)

    @staticmethod
    def default_mapper() -> LemmatizerBMP:
        return LemmatizerBMP(string.ascii_letters + string.digits + string.punctuation + string.whitespace + "\u2018\u2019\u201c\u201d\u201e\u00ab\u00bb\u2013\u2014\u2026")

    @staticmethod
    def default_categories(mapper: LemmatizerBMP) -> Tensor:
        # code 0 is the unknown character; code k is the k-th character of the mapper's alphabet
        def category(c: str) -> int:
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

    def tokenize(self, codes: Tensor, document_offsets: Optional[Tensor] = None) -> Tensor:
        # codes: (N,) of one or more concatenated texts, document_offsets: (D+1,) where each text starts in codes (no
        # token crosses a text boundary). Returns token spans (2, T): start and end (exclusive) positions in codes.
        n = codes.numel()
        if n == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=codes.device)
        category = self.categories.to(codes.device)[codes]                                          # (N,) uint8
        first = torch.zeros(n, dtype=torch.bool, device=codes.device)  # first character of a text
        if document_offsets is None:
            first[0] = True
        else:
            first[document_offsets[:-1][document_offsets[:-1] < n]] = True
        followed_by_simple = torch.zeros_like(first)
        followed_by_simple[:-1] = (category[1:] == SIMPLE) & ~first[1:]
        category[(category == ATTACHING) & ~followed_by_simple] = KEPT
        del followed_by_simple
        previous = torch.cat([category.new_full((1,), DUMPED), category[:-1]])
        previous[first] = DUMPED
        in_token = category != DUMPED
        starts = in_token & ((previous == DUMPED) | (category == ATTACHING) | ((category == SIMPLE) & (previous == KEPT))
                             | ((category == KEPT) & ((previous != KEPT) | (not self.merge_kept))))
        del previous, category
        ends = in_token.clone()  # last character of a token: the next one is outside any token or starts a new one
        ends[:-1] &= ~in_token[1:] | starts[1:] | first[1:]
        return torch.stack([starts.nonzero().reshape(-1), ends.nonzero().reshape(-1) + 1])

class Corpus(object):
    # A collection of texts as one sequence: their codes concatenated (no separators), each text's start as an offset,
    # tokens as spans into the sequence. Tokens are observed as their strings, lower-cased if `lowercase`; token_ids
    # index the corpus vocabulary, so documents are id tensors as the matchers need them.
    def __init__(self, texts: Iterable[str], tokenizer: Optional[Tokenizer] = None, lowercase: bool = True, device="cpu") -> None:
        self.texts = list(texts)
        self.tokenizer = Tokenizer() if tokenizer is None else tokenizer
        lengths = torch.tensor([len(t) for t in self.texts], dtype=torch.long)
        self.char_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)])
        codes = self.tokenizer.encode("".join(self.texts)).to(device)
        self.spans = self.tokenizer.tokenize(codes, self.char_offsets.to(device)).cpu()            # (2, T)
        del codes
        document = torch.searchsorted(self.char_offsets[1:], self.spans[0], right=True)              # (T,)
        self.token_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(torch.bincount(document, minlength=len(self.texts)), 0)])
        # Token strings one text at a time (never all at once): ids in order of first occurrence, then sorted.
        index: Dict[str, int] = {}
        ids = []
        for n, text in enumerate(self.texts):
            local = (self.spans[:, self.token_offsets[n]:self.token_offsets[n + 1]] - self.char_offsets[n]).T.tolist()
            strings = [text[start:end] for start, end in local]
            if lowercase:
                strings = [s.lower() for s in strings]
            ids.append(torch.tensor([index.setdefault(s, len(index)) for s in strings], dtype=torch.long))
        tokens = sorted(index, key=index.get)
        order = sorted(range(len(tokens)), key=tokens.__getitem__)
        self.vocabulary = Vocabulary([tokens[k] for k in order])
        rank = torch.empty(len(tokens), dtype=torch.long)
        rank[torch.tensor(order, dtype=torch.long)] = torch.arange(len(tokens))
        self.documents = [rank[document_ids] for document_ids in ids]                               # (tokens,) per text
        self.token_ids = torch.cat(self.documents) if self.documents else torch.zeros(0, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def char_span(self, document: int, start_token: int, end_token: int) -> Tuple[int, int]:
        # Tokens start_token .. end_token - 1 of a document -> their character span in its text.
        first = int(self.token_offsets[document]) + start_token
        last = int(self.token_offsets[document]) + end_token - 1
        offset = int(self.char_offsets[document])
        return int(self.spans[0, first]) - offset, int(self.spans[1, last]) - offset
