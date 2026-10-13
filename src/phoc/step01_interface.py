# Step 1 — the design API.
# PHOC is an abstract functor. This step fixes what every later step must honour: the constructor parameters,
# the inputs forward() accepts, the column layout of the embedding, the column weighting and the distance.
# Subclasses only implement embed_codes(); everything user-facing lives here.
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch
from pylelemmatize.fast_mapper import LemmatizerBMP
from torch import Tensor, nn


Words = str | Sequence[str] | tuple[Tensor, Tensor]


class PHOC(nn.Module, metaclass=ABCMeta):
    @staticmethod
    def default_mapper() -> LemmatizerBMP:
        alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        return LemmatizerBMP(mapping_dict={c: c.lower() for c in alphanumeric})

    def __init__(self, mapper: LemmatizerBMP | None = None, levels: Sequence[int] = (1, 2, 3, 5), occurrence: str = "count",
                 count_unknown: bool = False, level_weights: Sequence[float] | None = None, max_len: int = 20, len_weight: float = 1.0) -> None:
        super().__init__()
        if occurrence not in ("count", "presence"):
            raise ValueError(f"occurrence must be 'count' or 'presence', got {occurrence!r}")
        if len(levels) == 0 or any(level < 1 for level in levels):
            raise ValueError(f"levels must be a non-empty sequence of positive integers, got {levels!r}")
        if level_weights is None:
            level_weights = [1.0] * len(levels)
        if len(level_weights) != len(levels):
            raise ValueError(f"level_weights must have one weight per level, got {len(level_weights)} for {len(levels)} levels")
        if max_len < 0:
            raise ValueError(f"max_len must be non-negative, got {max_len}")
        self.mapper = self.default_mapper() if mapper is None else mapper
        # The mapper may merge characters (e.g. case folding) but its codes index the source alphabet;
        # re-encoding the mapped string over the destination alphabet gives the dense codes: 0 unknown, 1.. symbols.
        self.encoder = LemmatizerBMP(self.mapper.dst_alphabet_str)
        self.levels = tuple(int(level) for level in levels)
        self.occurrence = occurrence
        self.count_unknown = count_unknown
        self.level_weights = tuple(float(w) for w in level_weights)
        self.max_len = int(max_len)
        self.len_weight = float(len_weight)
        self.register_buffer("column_weights", self._create_column_weights(), persistent=False)

    @property
    def n_symbols(self) -> int:
        return len(self.encoder) if self.count_unknown else len(self.encoder) - 1

    @property
    def n_regions(self) -> int:
        return sum(self.levels)

    @property
    def dim(self) -> int:
        # Layout: for each level, for each of its regions, one column per symbol; then max_len thermometer length columns.
        return self.n_regions * self.n_symbols + self.max_len

    def _create_column_weights(self) -> Tensor:
        parts = [torch.full((level * self.n_symbols,), weight) for level, weight in zip(self.levels, self.level_weights)]
        parts.append(torch.full((self.max_len,), self.len_weight))
        return torch.cat(parts)

    def encode_strings(self, words: Sequence[str]) -> tuple[Tensor, Tensor]:
        lengths = torch.tensor([len(word) for word in words], dtype=torch.long)
        codes = self.encoder.str_to_intlabel_seq(self.mapper("".join(words)))
        codes = torch.from_numpy(codes.astype(np.int64))
        offsets = torch.zeros(len(words) + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        return codes, offsets

    @abstractmethod
    def embed_codes(self, codes: Tensor, offsets: Tensor) -> Tensor:
        # codes: (C,) dense symbol codes of all words concatenated; offsets: (B+1,) word b is codes[offsets[b]:offsets[b+1]].
        # Returns the unweighted (B, dim) embedding in the layout of dim.
        ...

    def forward(self, words: Words) -> Tensor:
        if isinstance(words, str):
            return self.forward([words])[0]
        if isinstance(words, tuple) and len(words) == 2 and all(isinstance(t, Tensor) for t in words):
            codes, offsets = words
        elif isinstance(words, (list, tuple)) and all(isinstance(word, str) for word in words):
            codes, offsets = self.encode_strings(words)
            codes, offsets = codes.to(self.column_weights.device), offsets.to(self.column_weights.device)
        else:
            raise TypeError("PHOC accepts a string, a sequence of strings, or a (codes, offsets) tuple of tensors")
        return self.embed_codes(codes, offsets) * self.column_weights

    @staticmethod
    def distance(a: Tensor, b: Tensor) -> Tensor:
        return torch.cdist(a, b, p=1)

    def extra_repr(self) -> str:
        return (f"levels={self.levels}, occurrence={self.occurrence!r}, count_unknown={self.count_unknown}, level_weights={self.level_weights}, "
                f"max_len={self.max_len}, len_weight={self.len_weight}, n_symbols={self.n_symbols}, dim={self.dim}")
