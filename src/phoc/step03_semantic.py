# Step 3 — from spelling to meaning.
# PHOCToSemantic maps a word's PHOC (step 2: fixed, learning-free) through a small trainable MLP into a semantic
# embedding space such as GloVe. Because PHOC is defined for any string, the mapping applies to words never seen in
# training and to misspelled or OCR-corrupted words; trained with corrupted inputs and clean targets (see
# train_semantic.py), it learns to see through the noise.
from typing import Any, Dict, List, Optional, Sequence

import torch
from pylelemmatize.fast_mapper import LemmatizerBMP
from torch import Tensor, nn

from .step02_overlap import PHOC


class PHOCToSemantic(nn.Module):
    def __init__(self, output_dim: int, hidden: Sequence[int] = (512, 512), levels: Sequence[int] = (1, 2, 3, 5), max_len: int = 20,
                 occurrence: str = "count", mapping: Optional[Dict[str, str]] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.settings = {"output_dim": output_dim, "hidden": list(hidden), "levels": list(levels), "max_len": max_len,
                         "occurrence": occurrence, "mapping": mapping, "dropout": dropout}
        mapper = LemmatizerBMP(mapping_dict=mapping) if mapping is not None else None
        self.phoc = PHOC(mapper=mapper, levels=levels, max_len=max_len, occurrence=occurrence)
        layers: List[nn.Module] = []
        width = self.phoc.dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, words: Sequence[str]) -> Tensor:
        # (B, output_dim), unit length: only directions matter in a cosine-similarity space.
        with torch.no_grad():
            phoc = self.phoc(list(words))
        return nn.functional.normalize(self.head(phoc), dim=1)

    def config(self) -> Dict[str, Any]:
        return dict(self.settings)

    @classmethod
    def from_checkpoint(cls, path: str, weights: str = "best", map_location: str = "cpu") -> "PHOCToSemantic":
        # weights: "best" (lowest validation loss so far) or "last" (end of the last completed epoch).
        state = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**state["model_config"])
        model.load_state_dict(state["best_model"] if weights == "best" else state["model"])
        return model.eval()
