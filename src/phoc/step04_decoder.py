# Step 4 — from PHOC back to a spelling.
# PHOCToString reads a word's spelling out of its PHOC with a character-level GRU: the PHOC (step 2) sets the GRU's
# initial state and conditions every step, and the GRU emits one character at a time until an end symbol. Trained on
# corrupted inputs and clean targets (see train_decoder.py) it becomes a spelling corrector that needs no dictionary,
# so it can also produce words it never saw; given a PHOC predicted from elsewhere (an image, a meaning) it spells it.
from typing import Any
from collections.abc import Sequence

import torch
from pylelemmatize.fast_mapper import LemmatizerBMP
from torch import Tensor, nn

from .step02_overlap import PHOC

END, START = 0, 1  # output symbols 0 and 1; the alphabet's characters follow


class PHOCToString(nn.Module):
    def __init__(self, alphabet: str, hidden: int = 512, embedding: int = 64, condition: int = 128, levels: Sequence[int] = (1, 2, 3, 5),
                 max_len: int = 20, max_output: int = 24, mapping: dict[str, str] | None = None, dropout: float = 0.1,
                 input_dropout: float = 0.0, input_noise: float = 0.0) -> None:
        # input_dropout / input_noise: training-time augmentation of the PHOC input — cells zeroed with that probability,
        # and Gaussian noise of that standard deviation added — so the decoder learns to read imperfect PHOCs.
        super().__init__()
        self.settings = {"alphabet": alphabet, "hidden": hidden, "embedding": embedding, "condition": condition, "levels": list(levels),
                         "max_len": max_len, "max_output": max_output, "mapping": mapping, "dropout": dropout,
                         "input_dropout": input_dropout, "input_noise": input_noise}
        mapper = LemmatizerBMP(mapping_dict=mapping) if mapping is not None else None
        self.phoc = PHOC(mapper=mapper, levels=levels, max_len=max_len)
        self.alphabet = alphabet
        self.index = {c: n + 2 for n, c in enumerate(alphabet)}
        self.initial = nn.Sequential(nn.Linear(self.phoc.dim, hidden), nn.Tanh())
        self.conditioning = nn.Sequential(nn.Linear(self.phoc.dim, condition), nn.ReLU())
        self.embed = nn.Embedding(len(alphabet) + 2, embedding)
        self.gru = nn.GRU(embedding + condition, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, len(alphabet) + 2)

    def config(self) -> dict[str, Any]:
        return dict(self.settings)

    def encode_targets(self, words: Sequence[str]) -> Tensor:
        # (B, L+1): the characters of each word then END, padded with END; characters outside the alphabet are skipped.
        codes = [[self.index[c] for c in w if c in self.index][:self.settings["max_output"] - 1] + [END] for w in words]
        out = torch.full((len(codes), max(len(c) for c in codes)), END, dtype=torch.long)
        for n, c in enumerate(codes):
            out[n, :len(c)] = torch.tensor(c)
        return out

    def _start(self, words: Sequence[str] | Tensor) -> tuple:
        # words: strings, or their PHOCs (B, phoc.dim) already computed or predicted (gradients then flow through them)
        if isinstance(words, Tensor):
            phoc = words
        else:
            with torch.no_grad():
                phoc = self.phoc(list(words))
        if self.training:
            phoc = nn.functional.dropout(phoc, p=self.settings["input_dropout"])
            phoc = phoc + self.settings["input_noise"] * torch.randn_like(phoc)
        return self.initial(phoc)[None], self.conditioning(phoc)  # (1, B, H), (B, C)

    def forward(self, words: Sequence[str] | Tensor, targets: Tensor) -> Tensor:
        # Teacher forcing: logits (B, L, symbols) for predicting targets[:, t] from targets[:, :t].
        hidden, condition = self._start(words)
        previous = torch.cat([torch.full((len(targets), 1), START, dtype=torch.long, device=targets.device), targets[:, :-1]], dim=1)
        inputs = torch.cat([self.embed(previous), condition[:, None].expand(-1, previous.size(1), -1)], dim=2)
        states, _ = self.gru(inputs, hidden)
        return self.output(self.dropout(states))

    @torch.no_grad()
    def spell(self, words: Sequence[str] | Tensor) -> list[str]:
        # Greedy decoding of the spelling encoded in each input's PHOC.
        hidden, condition = self._start(words)
        symbol = torch.full((len(words),), START, dtype=torch.long, device=condition.device)
        finished = torch.zeros(len(words), dtype=torch.bool, device=condition.device)
        spelled: list[list[str]] = [[] for _ in words]
        for _ in range(self.settings["max_output"]):
            states, hidden = self.gru(torch.cat([self.embed(symbol), condition], dim=1)[:, None], hidden)
            symbol = self.output(states[:, 0]).argmax(dim=1)
            finished |= symbol == END
            for n in (~finished).nonzero().reshape(-1).tolist():
                spelled[n].append(self.alphabet[int(symbol[n]) - 2] if int(symbol[n]) >= 2 else "")
            if finished.all():
                break
        return ["".join(characters) for characters in spelled]

    @classmethod
    def from_checkpoint(cls, path: str, weights: str = "best", map_location: str = "cpu") -> "PHOCToString":
        state = torch.load(path, map_location=map_location, weights_only=False)  # our own file: it holds RNG states and history
        model = cls(**state["model_config"])
        model.load_state_dict(state["best_model"] if weights == "best" else state["model"])
        return model.eval()
