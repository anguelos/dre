import torch
from typing import List, Dict
import nltk

from .wordre_parser import WordParser
from .abstract_embedder import AbstractEmbedder


class AbstractObserver(object):
    """A class that converts observations into a sequence of probabillities over an alphabet"""
    @property
    def observation_size(self) -> int:
        raise NotImplemented
    
    def __init__(self, embedder, regex_parser) -> None:
        self._embedder = embedder
        self._regex_parser = regex_parser

    def observe_sequence(self, tokens:List[str]) -> torch.Tensor:
        return torch.cat([self._observe_token(token)[:, None] for token in tokens], dim=1)

    def _observe_token(self, sequence:str) -> torch.Tensor:
        raise NotImplemented


class SequenceObserver(AbstractObserver):
    def __init__(self, embedder:AbstractEmbedder, regex_str:str, regex_parser:type=WordParser, any_symbol="ANY", null_symbol="", require_tokens=True, observer_b=10., device="cpu") -> None:
        super().__init__(embedder, regex_parser)
        self.observer_b = observer_b
        self.device = device
        self.t_regex_parser = type(regex_parser)
        self.fagraph = regex_parser.parse_str(regex_str)
        self.transitions, self.starting_states, self.accepting_states = self.fagraph.nfa_segment.create_transition_matrices()
        if require_tokens:
            for token in self.transitions.keys():
                assert token in embedder.keys()
        self.starting_states, self.accepting_states = torch.tensor(self.starting_states), torch.tensor(self.accepting_states)
        self.transitions.update({k:torch.tensor(v, device=device) for k, v in self.transitions.items()})
        self.null_symbol = null_symbol
        self.any_symbol = any_symbol
        
        null_transition = torch.ones_like(list(self.transitions.values())[0])  #  what would be a neutral value for cosine distance?
        self.transitions[self.null_symbol] = null_transition
        self.transition_symbols = sorted(tuple(self.transitions.keys()))
        self.transition_symbol_matrix = torch.cat([self._embedder[s][:,None] for s in self.transition_symbols], dim=1)
        self.all_transitions = torch.cat([self.transitions[s][None, :, :] for s in self.transition_symbols], dim=0)
        # all any transitions 
        self.any_symbol_distance = torch.ones(len(self.transitions), device=self.device)
        self.any_symbol_distance[0] = 0



    def get_observation_probabilities(self, token):
        if token == self.any_symbol:
            embedded_token = self._embedder[token][:, None]
        else:
            embedded_token = self._embedder[token][:, None]
            similarities = torch.cosine_similarity(self.transition_symbol_matrix, embedded_token, dim=0)
            shiftsig = lambda x,y: 1/(1+torch.exp(-(y*x)+y/2))
            return shiftsig(similarities, self.observer_b)
        #print(self.transition_symbol_matrix.size())
        #print(embedded_token.size())
        #similarities = torch.matmul( embedded_token, self.transition_symbol_matrix) # TODO (anguelos) what is the difference with torch.dot?
        #print(self.transition_symbol_matrix[:7,:])
        #print(self.transition_symbol_matrix[:7,:])
        #print(self.transition_symbol_matrix.size())
        #print(self.transition_symbols)
        #print(embedded_token.size())
        #print(similarities.size()) 
        #print("\n")
        #return torch.softmax((similarities*10), dim=0)


    def observe_transitions(self, tokens):
        #states = self.starting_states.copy()
        obs_probs = []
        for token in tokens:
            obs_probs.append(self.get_observation_probabilities(token)[None,:])
        return torch.cat(obs_probs, dim=0)
