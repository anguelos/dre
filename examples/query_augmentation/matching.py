# The dre matchers compared, each named <method>.<engine> (engine: viterbi, dfa). A method is an observer factory:
#   exact            tokens equal to the query word (after dre's default normalization)
#   phoc             soft PHOC, L1 distance (temperature 2, background 0.5 of the token's PHOC mass)
#   levenshtein      edit distance on character codes (temperature 0.22, background 0.27 edits per character)
#   cosine<dim>      GloVe <dim>d cosine distance (glove_soft's model: temperature 0.05, background 0.4)
#   csls<dim>        GloVe <dim>d CSLS distance (Conneau et al. 2018), hubs pushed away; same temperature/background
#   phoc_csls<dim>   the spelling chain spelling -> vocabulary word -> meaning, scored with CSLS: a token's row is the
#                    PHOC-softmax mix of the GloVe rows of the words it most probably spells, so misspelled tokens
#                    inherit the meaning of the words they resemble.
# GloVe rows are [unit vector, CSLS radius]; the radius of a word is its mean cosine to its 10 nearest neighbours among
# the 100k most frequent words, cached next to the vectors (stage.py's data/glove/glove<dim>.pt) as radius<dim>.pt.
# Nothing is tuned on the benchmarks.
import re
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

import dre

ENGINES = {"viterbi": dre.ViterbiMatcher, "dfa": dre.DFAMatcher}
GLOVE_METHOD = re.compile(r"(cosine|csls|phoc_csls)(\d+)")


def glove_rows(glove_root: str, dim: int, device="cpu", neighbours: int = 10, frequent: int = 100000,
               batch: int = 4096) -> tuple[list[str], Tensor]:
    # (words, rows): GloVe 6B words in frequency order, rows [unit vector, CSLS radius]
    state = torch.load(Path(glove_root) / f"glove{dim}.pt", weights_only=False)
    words, unit = state["words"], nn.functional.normalize(state["vectors"].float(), dim=1)
    cache = Path(glove_root) / f"radius{dim}.pt"
    if cache.exists():
        radius = torch.load(cache)
    else:
        work = unit.to(device)
        pool, radius = work[:frequent], torch.empty(len(words))
        for begin in range(0, len(words), batch):
            top = (work[begin:begin + batch] @ pool.T).topk(neighbours + 1, dim=1).values[:, 1:]   # without itself
            radius[begin:begin + batch] = top.mean(dim=1).cpu()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(radius, cache)
    return words, torch.cat([unit, radius[:, None]], dim=1)


class CosineDistance(dre.AbstractDistance):
    # 1 - cosine of the unit-vector parts; mass 1, so the background is a plain cosine distance. A row without a
    # vector (null row: an out-of-vocabulary token or query word) is at distance 2 from everything, beyond any
    # background, so an out-of-vocabulary word only matches identical tokens (the observer's same-form rule).
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.without_null(a, b, 1 - a[:, :self.dim] @ b[:, :self.dim].T)

    def without_null(self, a: Tensor, b: Tensor, distances: Tensor) -> Tensor:
        null_a, null_b = a[:, :self.dim].abs().sum(dim=1) == 0, b[:, :self.dim].abs().sum(dim=1) == 0
        return distances.masked_fill(null_a[:, None] | null_b[None, :], 2.0)

    def mass(self, a: Tensor) -> Tensor:
        return torch.ones(a.size(0), device=a.device)


class CSLSDistance(CosineDistance):
    # 1 - (cosine - (r_a + r_b) / 2 + mean radius): cosine units, words close to everything (hubs) pushed away
    def __init__(self, dim: int, mean_radius: float) -> None:
        super().__init__(dim)
        self.mean_radius = mean_radius

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        radii = (a[:, self.dim, None] + b[None, :, self.dim]) / 2
        return self.without_null(a, b, 1 - (a[:, :self.dim] @ b[:, :self.dim].T - radii + self.mean_radius))


class SpellingEmbedder(dre.AbstractEmbedder):
    # P(token spells v) = softmax(-PHOC L1 distance / temperature) over the token's `candidates` nearest of the
    # `vocabulary` most frequent GloVe words; the token's row is the mix of their rows (unit part renormalized). One
    # edit costs ~9 PHOC units, so a correctly spelled word keeps nearly all the mass on itself; a token GloVe knows
    # outside the frequent words is its own candidate at distance 0 (replacing the farthest one).
    def __init__(self, table: dre.TableEmbedder, dim: int, vocabulary: int = 50000, candidates: int = 20,
                 temperature: float = 2.0, batch: int = 2048, device="cpu") -> None:
        super().__init__(dre.default_mapping.identity_mapper())
        self.table, self.dim, self.candidates, self.temperature, self.batch = table, dim, candidates, temperature, batch
        self.words = list(table.index)[:vocabulary]
        self.pool = set(self.words)
        self.phoc = dre.PHOCEmbedder(device=device)
        self.word_phocs = torch.cat([self.phoc.embed(self.words[b:b + 8192]) for b in range(0, len(self.words), 8192)])
        self.word_rows = table.embed(self.words)

    @property
    def null_embedding(self) -> Tensor:
        return self.table.null_embedding

    def __getitem__(self, token: str) -> Tensor:
        return self.embed([token])[0]

    @torch.no_grad()
    def embed(self, tokens: list[str]) -> Tensor:
        if len(tokens) > self.batch:
            return torch.cat([self.embed(tokens[b:b + self.batch]) for b in range(0, len(tokens), self.batch)])
        if not tokens:
            return self.null_embedding.new_zeros([0, self.null_embedding.numel()])
        nearest, index = torch.cdist(self.phoc.embed(tokens), self.word_phocs, p=1).topk(self.candidates, dim=1, largest=False)
        rows = self.word_rows[index]                                                              # (N, C, row)
        outside = [n for n, token in enumerate(tokens) if token in self.table.index and token not in self.pool]
        if outside:
            at = torch.tensor(outside, device=nearest.device)
            rows[at, -1] = self.table.embed([tokens[n] for n in outside])
            nearest[at, -1] = 0
        mixed = (torch.softmax(-nearest / self.temperature, dim=1)[:, :, None] * rows).sum(dim=1)
        mixed[:, :self.dim] = nn.functional.normalize(mixed[:, :self.dim], dim=1)
        return mixed


def observer_factory(method: str, classes: dre.Classes | None = None, glove_root: str = "data/glove",
                     device="cpu") -> Callable[[dre.Vocabulary], dre.SequenceObserver]:
    # method without the engine suffix
    if method == "exact":
        return lambda vocabulary: dre.ExactObserver(vocabulary, classes=classes)
    if method == "phoc":
        embedder, distance, temperature, background = dre.PHOCEmbedder(device=device), dre.L1Distance(), 2.0, 0.5
    elif method == "levenshtein":
        embedder, distance, temperature, background = dre.CharCodeEmbedder(32, device=device), dre.LevenshteinDistance(), 0.22, 0.27
    elif GLOVE_METHOD.fullmatch(method):
        kind, dim = GLOVE_METHOD.fullmatch(method).groups()
        dim = int(dim)
        words, rows = glove_rows(glove_root, dim, device)
        embedder = dre.TableEmbedder(words, rows, device=device)
        distance = CosineDistance(dim) if kind == "cosine" else CSLSDistance(dim, float(rows[:, dim].mean()))
        if kind == "phoc_csls":
            embedder = SpellingEmbedder(embedder, dim, device=device)
        temperature, background = 0.05, 0.4
    else:
        raise ValueError(f"unknown method {method!r}")
    return lambda vocabulary: dre.JointSoftmaxObserver(vocabulary, embedder, distance, classes, temperature=temperature,
                                                       background=background, device=device)


def split_method(name: str) -> tuple[str, str]:
    # "phoc_csls100.viterbi" -> ("phoc_csls100", "viterbi")
    method, _, engine = name.partition(".")
    if engine not in ENGINES:
        raise ValueError(f"{name!r}: expected <method>.<engine>, engine one of {sorted(ENGINES)}")
    return method, engine
