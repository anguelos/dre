# Step 2 — the PHOC computation.
# Character i of an n-character word occupies [i/n, (i+1)/n]; region r of level L covers [r/L, (r+1)/L].
# The share of the character that falls inside the region is what it contributes to that region's histogram:
#   "count"    sums the shares, so a character straddling two regions counts half (or a third…) in each;
#   "presence" marks a symbol present if at least half of one of its characters lies in the region (Almazán et al.).
# Shares are computed in integer units of 1/(n·L), so they are exact and the 1/2 threshold has no rounding issues.

import torch
from torch import Tensor

from .step01_interface import PHOC as PHOCInterface


class PHOC(PHOCInterface):
    def _regions(self, device: torch.device) -> tuple[Tensor, Tensor]:
        levels = torch.tensor([level for level in self.levels for _ in range(level)], device=device)
        indexes = torch.tensor([r for level in self.levels for r in range(level)], device=device)
        return levels, indexes

    def embed_codes(self, codes: Tensor, offsets: Tensor) -> Tensor:
        device = codes.device
        lengths = offsets[1:] - offsets[:-1]
        n_words = lengths.numel()
        word = torch.repeat_interleave(torch.arange(n_words, device=device), lengths)  # (C,) word of each character
        i = (torch.arange(codes.numel(), device=device) - offsets[:-1][word])[:, None]  # (C, 1) position in its word
        n = lengths[word][:, None]                                                        # (C, 1) length of its word
        level, r = self._regions(device)                                                  # (R,), (R,)

        # Character [i·L, (i+1)·L] and region [r·n, (r+1)·n], both in units of 1/(n·L).
        overlap = (torch.minimum((i + 1) * level, (r + 1) * n) - torch.maximum(i * level, r * n)).clamp(min=0)  # (C, R)
        if self.occurrence == "count":
            share = overlap / level
        else:
            share = (2 * overlap >= level).float()

        column = codes if self.count_unknown else codes - 1
        known = column >= 0
        cell = (word[known, None] * self.n_regions + torch.arange(self.n_regions, device=device)) * self.n_symbols + column[known, None]
        histogram = torch.zeros(n_words * self.n_regions * self.n_symbols, device=device)
        histogram.index_add_(0, cell.reshape(-1), share[known].reshape(-1).float())
        histogram = histogram.view(n_words, -1)
        if self.occurrence == "presence":
            histogram = histogram.clamp(max=1)

        length = (lengths[:, None] >= torch.arange(1, self.max_len + 1, device=device)).float()  # thermometer
        return torch.cat([histogram, length], dim=1)
