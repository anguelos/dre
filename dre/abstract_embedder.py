import torch
from typing import List, Dict

from .wordre_parser import WordParser
from .symbolre_parser import WordParser



class AbstractEmbedder(object):
    @property
    def embedding_size(self):
        return self.null_embedding.size(0)
    @property
    def null_embedding(self) -> torch.Tensor:
        raise NotImplemented

    def __getitem__(self, token:str) -> torch.Tensor:
        raise NotImplemented


class SymbolEmbedder(AbstractEmbedder):
    def __init__(self, values:Dict[str, int]):
        self.num_values = {}
        self.num_values.update(values)
        assert '' in self.values.keys()
        self.null_value = self.values['']
        self.embeddings = {}
        for symbol, num in self.num_values.items:
            embedding = torch.zeros(len(self(values)))
            embedding[num] = 1
            self.embeddings[symbol] = embedding
        self._null_embedding = self.embeddings['']

    def __getitem__(self, token: str) -> torch.Tensor:
        return self.embeddings.get(token, self._null_embedding)
    
    @property
    def null_embedding(self) -> torch.Tensor:
        return self._null_embedding


class Glove50Embedder(AbstractEmbedder):
    avg = torch.tensor([ 0.072617, -0.51393 ,  0.4728  , -0.52202 , -0.35534 ,  0.34629 ,
        0.23211 ,  0.23096 ,  0.26694 ,  0.41028 ,  0.28031 ,  0.14107 ,
       -0.30212 , -0.21095 , -0.10875 , -0.33659 , -0.46313 , -0.40999 ,
        0.32764 ,  0.47401 , -0.43449 ,  0.19959 , -0.55808 , -0.34077 ,
        0.078477,  0.62823 ,  0.17161 , -0.34454 , -0.2066  ,  0.1323  ,
       -1.8076  , -0.38851 ,  0.37654 , -0.50422 , -0.012446,  0.046182,
        0.70028 , -0.010573, -0.83629 , -0.24698 ,  0.6888  , -0.17986 ,
       -0.066569, -0.48044 , -0.55946 , -0.27594 ,  0.056072, -0.18907 ,
       -0.59021 ,  0.55559 ])
    
    @staticmethod
    def load_glove(path="tmp/glove/glove.6B.50d.txt"):
        lines = [l.split() for l in open(path, "r").read().strip().split("\n")]
        embbeding_list = [(l[0], torch.tensor([float(c) for c in l[1:]])) for l in lines]
        del lines
        embbeding_dict = {}
        for word, embeeding in embbeding_list:
            embbeding_dict[word] = torch.tensor(embeeding)
        return embbeding_dict, embbeding_list
    
    @property
    def null_embedding(self) -> torch.Tensor:
        return Glove50Embedder.avg.size(0)

    def __init__(self, glove_path="tmp/glove/glove.6B.50d.txt") -> None:
        self.embbeding_dict, self.embbeding_list = Glove50Embedder.load_glove(glove_path)

    def __getitem__(self, token):
        return self.embbeding_dict.get(token,Glove50Embedder.avg)


class AbstractObserver(object):
    @property
    def observation_size(self) -> int:
        raise NotImplemented

    def observe_sequence(self, sequence:str) -> torch.Tensor:
        tokens = self._tokenize_sequence(sequence)
        return torch.cat([self._observe_token(token)[:, None] for token in tokens], dim=1)

    def _tokenize_sequence(self, sequence:str) -> List[str]:
        raise NotImplemented

    def _observe_token(self, sequence:str) -> torch.Tensor:
        raise NotImplemented


class SequenceObserver(object):
    def __init__(self, embeder:AbstractEmbedder, regex:str, t_regex_parser:type=WordParser) -> None:
        super().__init__()
        self.embeder = embeder
        self.t_regex_parser = t_regex_parser
        self.fagraph = t_regex_parser().parse_str(regex)
        self.transitions, self.starting_states, self.accepting_states = self.graph.nfa_segment.create_transition_matrices()
        self.null_symbol = ''
        null_transition = torch.zeros_like(self.transitions.values[0])
        self.transitions[self.null_symbol] = null_transition
        self.transition_symbols = sorted(tuple(self.transitions.keys())+('',))
        self.all_transitions = torch.cat([self.transitions[s][None, :, :] for s in self.transition_symbols], dim=0)
    
    def get_observation_probabilities(self, token):
        raise NotImplemented

    def observe_transitions(self, tokens):
        states = self.starting_states.copy()
        for token in tokens:
            obs_probs = self.get_observation_probabilities(token)[None,:]


        