import torch
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from .abstract_embedder import AbstractEmbedder
from .abstract_distance import AbstractDistance
from .fa_graph import ANY, Automaton, is_class
from .tokenizing import Vocabulary

Classes = Dict[str, Callable[[str], bool]]  # token class symbol, e.g. "<number>", -> membership test for a token


def query_words(automaton: Automaton) -> List[str]:
    return [symbol for symbol in automaton.symbols if symbol != ANY and not is_class(symbol)]


class AbstractObserver(object):
    """A class that converts observations into a sequence of probabillities over an alphabet

    The alphabet is an automaton's symbols: observing tokens gives a (tokens, symbols) table of the probability that
    each token is each symbol. ANY (".") is every token with probability 1 and a token class "<name>" is exactly the
    tokens for which classes["<name>"] holds; the probability of being a query word comes from the token-to-word
    distances (embedder + distance) through _word_probabilities, which each technique implements.
    Without an embedder, a token is at distance 0 from the words it equals (case-insensitively) and 1 from the others.
    """
    def __init__(self, embedder: Optional[AbstractEmbedder] = None, distance: Optional[AbstractDistance] = None,
                 classes: Optional[Classes] = None, batch_size: int = 65536, device="cpu") -> None:
        self._embedder = embedder
        self._distance = AbstractDistance() if distance is None else distance
        self.classes = {} if classes is None else classes
        self.batch_size = batch_size
        self.device = device

    def observe_sequence(self, automaton: Automaton, tokens: List[str]) -> torch.Tensor:
        # (T, K): the tokens of one sequence, nothing cached
        distances, masses = self._distances(tokens, query_words(automaton))
        return self._symbol_table(automaton, tokens, distances, masses)

    def _word_probabilities(self, distances: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        # distances: (N, W) between N tokens and an automaton's W query words, masses: (N,) -> probabilities (N, W)
        raise NotImplementedError

    def _distances(self, tokens: List[str], words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # (N, W) distances and (N,) masses, on the CPU; tokens are embedded batch_size at a time on self.device and only
        # the distances are kept. Identical strings are at distance 0 whatever the embedding (e.g. a token without a
        # vector, which would otherwise be far from itself).
        distances = torch.ones(len(tokens), len(words))
        masses = torch.ones(len(tokens))
        word_index = {word: k for k, word in enumerate(words)}
        if self._embedder is None:
            for n, token in enumerate(tokens):
                if token.lower() in word_index:
                    distances[n, word_index[token.lower()]] = 0
            return distances, masses
        query = self._embedder.embed(words).to(self.device) if words else None
        for begin in range(0, len(tokens), self.batch_size):
            embedded = self._embedder.embed(tokens[begin:begin + self.batch_size]).to(self.device)
            if query is not None:
                distances[begin:begin + self.batch_size] = self._distance(embedded, query).cpu()
            masses[begin:begin + self.batch_size] = self._distance.mass(embedded).cpu()
        for n, token in enumerate(tokens):
            if token in word_index:
                distances[n, word_index[token]] = 0
        return distances, masses

    def _symbol_table(self, automaton: Automaton, tokens: List[str], distances: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        # distances: (N, W) for the automaton's query_words in their order
        table = torch.zeros(len(tokens), len(automaton.symbols))
        for k, symbol in enumerate(automaton.symbols):
            if symbol == ANY:
                table[:, k] = 1
            elif is_class(symbol):
                if symbol not in self.classes:
                    raise ValueError(f"Token class {symbol} needs a membership test in the observer's classes")
                table[:, k] = torch.tensor([float(self.classes[symbol](token)) for token in tokens])
        words = query_words(automaton)
        if words:
            table[:, [automaton.symbols.index(word) for word in words]] = self._word_probabilities(distances, masses)
        return table


class SequenceObserver(AbstractObserver):
    """Observes a fixed vocabulary (the distinct tokens of a corpus) for any number of automata.

    observe(automaton) gives the (V, K) table of the whole vocabulary; a document, encoded as token ids of the
    vocabulary, reads its rows. Distances between the vocabulary and query words are cached per word, so a word shared
    by many regexes is computed once, and prepare(words) computes the words of many regexes in one pass over the
    vocabulary. The base implementation is exact matching (probability 1 iff the distance is 0).
    """
    def __init__(self, vocabulary: Union[Vocabulary, List[str]], embedder: Optional[AbstractEmbedder] = None,
                 distance: Optional[AbstractDistance] = None, classes: Optional[Classes] = None, batch_size: int = 65536,
                 device="cpu") -> None:
        super().__init__(embedder, distance, classes, batch_size, device)
        self.vocabulary = vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary(vocabulary)
        self._word_columns: Dict[str, torch.Tensor] = {}
        self._masses: Optional[torch.Tensor] = None

    def prepare(self, words: Iterable[str]) -> None:
        new_words = [word for word in dict.fromkeys(words) if word not in self._word_columns]
        if not new_words and self._masses is not None:
            return
        distances, self._masses = self._distances(self.vocabulary.tokens, new_words)
        self._word_columns.update({word: distances[:, k] for k, word in enumerate(new_words)})

    def observe(self, automaton: Automaton) -> torch.Tensor:
        # (V, K)
        words = query_words(automaton)
        self.prepare(words)
        distances = torch.stack([self._word_columns[word] for word in words], dim=1) if words else torch.zeros(len(self.vocabulary), 0)
        return self._symbol_table(automaton, self.vocabulary.tokens, distances, self._masses)

    def _word_probabilities(self, distances: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        return (distances == 0).float()


class ExactObserver(SequenceObserver):
    # A token is a query word iff they are the same string (case-insensitively, without an embedder) or at distance 0.
    pass


class JointSoftmaxObserver(SequenceObserver):
    # P(token is word k) is a softmax over the automaton's words plus one background class "some other word":
    #     logit_k = -d(token, word_k) / temperature        logit_background = -background * mass(token) / temperature
    # A token is as likely to be word k as something else when its distance to k equals `background` times its own
    # mass, so the threshold scales with the token (word length for PHOC and Levenshtein); `temperature` sets how
    # sharply the probability falls around it. The query words compete for each token: a token close to two of them
    # splits its probability between them.
    def __init__(self, vocabulary: Union[Vocabulary, List[str]], embedder: Optional[AbstractEmbedder] = None,
                 distance: Optional[AbstractDistance] = None, classes: Optional[Classes] = None, temperature: float = 2.0,
                 background: float = 0.5, batch_size: int = 65536, device="cpu") -> None:
        super().__init__(vocabulary, embedder, distance, classes, batch_size, device)
        self.temperature = temperature
        self.background = background

    def _word_probabilities(self, distances: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        logits = -torch.cat([distances, self.background * masses[:, None]], dim=1) / self.temperature
        return torch.softmax(logits, dim=1)[:, :-1]  # drop the background


class SigmoidObserver(SequenceObserver):
    # Each word on its own: P(token is word k) = sigmoid(b * (threshold - d(token, word_k))). With CosineDistance and
    # threshold 0.5 this is the original observer's shifted sigmoid of the cosine similarity, 1 / (1 + exp(-b s + b / 2)).
    def __init__(self, vocabulary: Union[Vocabulary, List[str]], embedder: Optional[AbstractEmbedder] = None,
                 distance: Optional[AbstractDistance] = None, classes: Optional[Classes] = None, b: float = 10.,
                 threshold: float = .5, batch_size: int = 65536, device="cpu") -> None:
        super().__init__(vocabulary, embedder, distance, classes, batch_size, device)
        self.b = b
        self.threshold = threshold

    def _word_probabilities(self, distances: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.b * (self.threshold - distances))
