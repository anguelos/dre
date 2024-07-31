from collections import defaultdict
from typing import Dict, List, Union
import torch
from torch.nn import functional as F


class Alphabet:
    @staticmethod
    def load_alphabet_from_string(s: str):
        pass
    def __init__(self, code2symbol, symbol2code, ):
        pass
    def __repr__(self):
        pass
    def __get_item__(self, v):
        pass
    def encode_symbol(self, s:str)->int:
        pass
    def decode_symbol(self, s:int)->str:
        pass
    def encode_sequence(self, s: str)->torch.Tensor:
        pass
    def __len__(self):
        pass


def seq2oh(seq: str, alphabet: str) -> torch.Tensor:
    """Convert a sequence of characters to a one-hot tensor"""
    oh = torch.zeros(len(alphabet), len(seq))
    for i, s in enumerate(seq):
        oh[alphabet.index(s), i] = 1
    return oh


def oh2seq(oh: torch.Tensor, alphabet: str) -> str:
    """Convert a one-hot tensor to a sequence of characters"""
    return "".join([alphabet[torch.argmax(oh[:, i]).item()] for i in range(oh.size(1))])


def get_single_phoc_slices(tc_seq: torch.Tensor, split_count):
    ct_seq = tc_seq.swapaxes(0, 1)
    res = F.avg_pool1d(input=ct_seq, split_count)
    return res.view([1, 1, -1])

def get_single_phoc(btc_seq: torch.Tensor, pyramid: List[int]):
    layers = []
    for splits in pyramid:
        layers.append(get_single_phoc_slices(btc_seq, splits))
    return torch.cat(layers, dim=2)


class PHOC(torch.nn.modules.Module):
    def __init__(self, alphabet: Union[str, Dict[str, int]], levels: List[int]):
        super().__init__()
        if isinstance(alphabet, str):
            self.alphabet = alphabet
            unigrams = {c: n + 1 for n, c in enumerate(alphabet)}
            self.symbol2int = defaultdict(lambda: 0)
            self.symbol2int.update(unigrams)
        elif isinstance(alphabet, dict):
            assert 0 not in alphabet.values(), "Unigrams must start at 1"
            assert set(alphabet.values()) == set(range(1, len(alphabet) + 1)), "Unigrams must be consecutive"
            assert all([len(c) == 1 for c in alphabet]), "Unigrams must contain single characters" # TODO: Maybe allow ... etc
            assert len(set(alphabet.keys())) == len(alphabet), "Unigrams must be unique"
            assert all([isinstance(c, str) for c in alphabet]), "Unigrams must be strings"
            assert all([isinstance(n, int) for n in alphabet.values()]), "Unigram codes must be integers"
            self.symbol2int = defaultdict(lambda: 0)
            self.symbol2int.update(alphabet)
            items = sorted([(n, c) for c, n in alphabet.items()])
            self.alphabet = "".join([c for _, c in items])
        self.max_code = max(self.symbol2int.values())
        self.pooling_layers = torch.nn.ModuleList([torch.nn.AdaptiveAvgPool1d(n) for n in levels])
    
    def forward_single_bct(self, bct_seq: torch.Tensor) -> torch.Tensor:
        """Compute the PHOC of a sequence"""
        pyramid_layers = []
        for pool in self.pooling_layers:
            pyramid_layers.append(pool(bct_seq).view([1, 1, -1]))
        return torch.cat(pyramid_layers, dim=2)
    
    def str2oh(self, seq: str) -> torch.Tensor:
        """Convert a sequence of characters to a one-hot tensor"""
        oh = torch.zeros(self.max_code + 1, len(seq))
        for i, s in enumerate(seq):
            oh[self.alphabet.index(s), i] = 1
        return oh[None, :, :]
    
    def forward_sample(self, word: Union[str, torch.Tensor], sentence_lengths: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """Compute the PHOC of a sequence"""
        if isinstance(word, str):
            bct_oh = self.str2oh(word)
        elif isinstance(word, torch.Tensor) and word.dim() == 3 and word.size(0) == 1 and word.size(1) == len(self.alphabet):
            btc_oh = word
            return self.forward_single_bct(btc_oh)
        elif isinstance(word, torch.Tensor) and word.dim() == 3 and word.size(0) > 1 and word.size(1) == len(self.alphabet):
            assert sentence_lengths is not None and sentence_lengths.dim() == 1 and sentence_lengths.size(0) == word.size(0)
            phocs = []
            for n in range(word.size(0)):
                btc_oh = word[n: n + 1, :, :sentence_lengths[n]]
                phoc = self.forward_single_bct(btc_oh)
                if n == 0:
                    res = phoc
                else:
                    res = torch.cat([res, phoc], dim=0)
            btc_oh = word
            return self.forward_single_bct(btc_oh)
        else:
            raise ValueError("seq must be a string or a tensor")
        return self.forward_single_bct(oh)

    def forward_batch(self, words: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Compute the PHOC of a batch of sequences"""
        if isinstance(words, list):
            res = torch.empty([self.])