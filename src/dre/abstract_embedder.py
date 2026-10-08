import torch
from typing import List, Dict, Optional

from phoc import PHOC


class AbstractEmbedder(object):
    """A class that converts tokens to embeddings.
    """
    @property
    def embedding_size(self) -> int:
        return self.null_embedding.size(0)

    @property
    def null_embedding(self) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, token:str) -> torch.Tensor:
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def embed(self, tokens:List[str]) -> torch.Tensor:
        # (N, embedding_size); subclasses that can embed many tokens at once override this
        if len(tokens) == 0:
            return self.null_embedding.new_zeros([0, self.embedding_size])
        return torch.stack([self[token] for token in tokens])


class TableEmbedder(AbstractEmbedder):
    # A fixed table of vectors; tokens outside it get the null embedding.
    def __init__(self, words:List[str], vectors:torch.Tensor, null_embedding:Optional[torch.Tensor]=None, device="cpu"):
        self.device = device
        self.index = {word: n for n, word in enumerate(words)}
        null_embedding = torch.zeros(vectors.size(1)) if null_embedding is None else null_embedding
        self.vectors = torch.cat([vectors, null_embedding[None, :].to(vectors.dtype)]).to(device)  # last row: null
        self._null_embedding = self.vectors[-1]

    @property
    def null_embedding(self) -> torch.Tensor:
        return self._null_embedding

    def keys(self):
        return self.index.keys()

    def __getitem__(self, token: str) -> torch.Tensor:
        return self.vectors[self.index.get(token, -1)]

    def embed(self, tokens:List[str]) -> torch.Tensor:
        rows = torch.tensor([self.index.get(token, len(self.index)) for token in tokens], dtype=torch.long, device=self.vectors.device)
        return self.vectors[rows]


class SymbolEmbedder(TableEmbedder):
    # One-hot vectors; values maps every symbol, including the null symbol '', to its dimension.
    def __init__(self, values:Dict[str, int], device="cpu"):
        assert '' in values.keys()
        self.num_values = dict(values)
        symbols = list(self.num_values.keys())
        vectors = torch.zeros(len(symbols), len(symbols))
        vectors[torch.arange(len(symbols)), torch.tensor([self.num_values[s] for s in symbols])] = 1
        super().__init__(symbols, vectors, null_embedding=vectors[symbols.index('')], device=device)


class Glove50Embedder(TableEmbedder):
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
    def load_glove(path:str="tmp/glove/glove.6B.50d.txt", device="cpu") :
        lines = [l.split() for l in open(path, "r").read().strip().split("\n")]
        embbeding_list = [(l[0], torch.tensor([float(c) for c in l[1:]], device=device)) for l in lines]
        del lines
        embbeding_dict = {}
        for word, embeeding in embbeding_list:
            embbeding_dict[word] = embeeding.clone().detach()
        return embbeding_dict, embbeding_list

    def __init__(self, glove_path:str="tmp/glove/glove.6B.50d.txt", device="cpu") -> None:
        self.embbeding_dict, self.embbeding_list = Glove50Embedder.load_glove(glove_path)
        words = list(self.embbeding_dict.keys())
        super().__init__(words, torch.stack([self.embbeding_dict[w] for w in words]), Glove50Embedder.avg, device=device)


class PHOCEmbedder(AbstractEmbedder):
    # Open vocabulary: any string has a PHOC, so there are no keys and no token gets the null embedding.
    def __init__(self, phoc:Optional[PHOC]=None, device="cpu"):
        self.phoc = (PHOC() if phoc is None else phoc).to(device)
        self._null_embedding = torch.zeros(self.phoc.dim, device=device)

    @property
    def null_embedding(self) -> torch.Tensor:
        return self._null_embedding

    def __getitem__(self, token: str) -> torch.Tensor:
        return self.embed([token])[0]

    def embed(self, tokens:List[str]) -> torch.Tensor:
        if len(tokens) == 0:
            return self._null_embedding.new_zeros([0, self.phoc.dim])
        with torch.no_grad():
            return self.phoc(list(tokens))


class CharCodeEmbedder(AbstractEmbedder):
    # A token as its unicode code points, zero-padded to max_length (longer tokens are truncated): the "embedding"
    # string distances such as Levenshtein work on.
    def __init__(self, max_length:int=32, device="cpu"):
        self.max_length = max_length
        self._null_embedding = torch.zeros(max_length, dtype=torch.long, device=device)

    @property
    def null_embedding(self) -> torch.Tensor:
        return self._null_embedding

    def __getitem__(self, token: str) -> torch.Tensor:
        return self.embed([token])[0]

    def embed(self, tokens:List[str]) -> torch.Tensor:
        codes = torch.zeros(len(tokens), self.max_length, dtype=torch.long)
        for n, token in enumerate(tokens):
            token_codes = [ord(c) for c in token[:self.max_length]]
            codes[n, :len(token_codes)] = torch.tensor(token_codes, dtype=torch.long)
        return codes.to(self._null_embedding.device)
