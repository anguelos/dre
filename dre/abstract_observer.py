"""Observers are defined here.

    Observers are classes that convert sequences of tokens into a sequence of vectors
    meant to be intepreted as probabilities over an alphabet/vocabulary.

"""
from abc import ABC, abstractmethod
import torch
from .typedefs import t_device, List
from .wordre_parser import WordParser
from .abstract_embedder import AbstractEmbedder


class AbstractObserver(ABC):
    """A class that converts observations into a sequence of probabillities over an alphabet"""
    @property
    def observation_size(self) -> int:
        """The number of symbols beeing observed.

        Returns:
            int: the number of symbols beeing observed.
        """
        return self._get_observation_size()

    @abstractmethod
    def _get_observation_size(self) -> int:
        """Abstract method used for the observation size property

        Returns:
            int: the size of the observation
        """

    def __init__(self, embedder, regex_parser) -> None:
        self._embedder = embedder
        self._regex_parser = regex_parser

    def observe_sequence(self, tokens: List[str]) -> torch.Tensor:
        return torch.cat([self._observe_token(token)[:, None] for token in tokens], dim=1)

    @abstractmethod
    def _observe_token(self, token: str) -> torch.Tensor:
        """Returns the probabillities of a symbol beeing observed.

        Args:
            token (str): the symbol(token).

        Returns:
            torch.Tensor: the probabillity the token was any of the monitored symbols.
        """


class SequenceObserver(AbstractObserver):
    def __init__(self, embedder: AbstractEmbedder, regex_str: str, regex_parser: WordParser,
                 any_symbol: str = "ANY", null_symbol: str = "", require_tokens: bool = True,
                 observer_b: float = 10., device: t_device = "cpu") -> None:
        super().__init__(embedder, regex_parser)
        self.observer_b = observer_b
        self.device = device
        self.t_regex_parser = type(regex_parser)
        self.fagraph = regex_parser.parse_str(regex_str)
        self.transitions, self.starting_states, self.accepting_states = self.fagraph.nfa_segment.create_transition_matrices()
        if require_tokens:
            for token in self.transitions.keys():
                assert token in embedder.keys()
        self.starting_states, self.accepting_states = torch.tensor(self.starting_states, device=device), torch.tensor(self.accepting_states, device=device)
        self.transitions.update({k: torch.tensor(v, device=device) for k, v in self.transitions.items()})
        self.null_symbol = null_symbol
        self.any_symbol = any_symbol

        null_transition = torch.ones_like(list(self.transitions.values())[0])  # what would be a neutral value for cosine distance?
        self.transitions[self.null_symbol] = null_transition
        self.transition_symbols = sorted(tuple(self.transitions.keys()))
        self.transition_symbol_matrix = torch.cat([self._embedder[s][:, None] for s in self.transition_symbols], dim=1)
        self.all_transitions = torch.cat([self.transitions[s][None, :, :] for s in self.transition_symbols], dim=0)
        # all any transitions
        self.any_symbol_distance = torch.ones(len(self.transitions), device=self.device)
        self.any_symbol_distance[0] = 0

    def _observe_token(self, token: str) -> torch.Tensor:
        # TODO(anguelos) verify any_symbol is not compared agains tokens but symbols in the regex
        # if token == self.any_symbol:
        #    embedded_token = self._embedder[token][:, None]
        # else:
        embedded_token = self._embedder[token][:, None]
        similarities = torch.cosine_similarity(self.transition_symbol_matrix, embedded_token, dim=0)
        return similarities

    def _get_observation_size(self) -> int:
        return len(self.transition_symbols)

    def get_observation_probabilities(self, token: str) -> torch.Tensor:
        def shiftsig(x, y):
            return 1/(1 + torch.exp(-(y * x) + y / 2))
        similarities = self._observe_token(token)[:, None]
        return shiftsig(similarities, self.observer_b)

    def observe_sequence(self, tokens: List[str]) -> torch.Tensor:
        """Takes a list of tokens and returns a tensor of having observed tokens from the vocabulary.

        Args:
            tokens (List[str]): a sequence of tokens.

        Returns:
            torch.Tensor: a pdf each token belongs to the vocabulary.
        """
        obs_probs = []
        for token in tokens:
            obs_probs.append(self.get_observation_probabilities(token)[None, :])
        return torch.cat(obs_probs, dim=0)

    def match_transitions(self, tokens: List[str]) -> torch.Tensor:
        """Matches a sequence of tokens to the regular expression.

        Args:
            tokens (List[str]): the sequence of tokens to observed

        Returns:
            torch.Tensor: the probability of the sequence of tokens being accepted by the regular expression.
        """
        state = self.starting_states.clone()
        transitions = self.observe_sequence(tokens)
        state2state = self.all_transitions
        outputs = torch.zeros([state.size(0), len(tokens)])
        for n in range(len(tokens)):
            all_states = (state2state * state[None, :, None]).sum(dim=2)  # * (transitions[n,:, None])).sum(dim=0)
            state = (all_states * (transitions[n, :]))
            state = state.sum(dim=0)
            outputs[:, n] = state
        return (outputs * self.accepting_states[:, None]).max(dim=0)

