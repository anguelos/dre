# Step 5 — from meaning back to spelling.
# GloveToPHOC maps a semantic embedding (e.g. a GloVe vector) to the PHOC of the word it stands for, with a small MLP
# whose output is non-negative like a PHOC. With PHOCToSemantic (step 3) it closes a cycle — spelling -> meaning ->
# spelling — and with PHOCToString (step 4) a vector can be spelled out: phoc2str(glove2phoc(v)). Trained jointly with
# both, cycle-consistently (examples/phoc/train.py).
from typing import Any
from collections.abc import Sequence

import torch
from pylelemmatize.fast_mapper import LemmatizerBMP
from torch import Tensor, nn

from .step02_overlap import PHOC


class GloveToPHOC(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int] = (512, 512), levels: Sequence[int] = (1, 2, 3, 5), max_len: int = 20,
                 occurrence: str = "count", mapping: dict[str, str] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.settings = {"input_dim": input_dim, "hidden": list(hidden), "levels": list(levels), "max_len": max_len,
                         "occurrence": occurrence, "mapping": mapping, "dropout": dropout}
        mapper = LemmatizerBMP(mapping_dict=mapping) if mapping is not None else None
        self.phoc = PHOC(mapper=mapper, levels=levels, max_len=max_len, occurrence=occurrence)  # the targets' layout
        layers: list[nn.Module] = []
        width = input_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers += [nn.Linear(width, self.phoc.dim), nn.ReLU()]  # PHOCs are non-negative
        self.head = nn.Sequential(*layers)

    def forward(self, vectors: Tensor) -> Tensor:
        # (B, input_dim) -> predicted PHOCs (B, phoc.dim); inputs are used as unit vectors, as in PHOCToSemantic's space
        return self.head(nn.functional.normalize(vectors, dim=1))

    @torch.no_grad()
    def targets(self, words: Sequence[str]) -> Tensor:
        # the PHOCs the predictions are trained to match
        return self.phoc(list(words))

    def config(self) -> dict[str, Any]:
        return dict(self.settings)
