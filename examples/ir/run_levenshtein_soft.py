#!/usr/bin/env python3
# Method "levenshtein_soft": phoc_soft with the PHOC L1 distance replaced by the Levenshtein (edit) distance between
# document-vocabulary tokens and query words; everything else — joint softmax, background, engine — is unchanged.
# It isolates what PHOC contributes beyond word-by-word probabilistic matching.
# A token's "mass" (its background scale) is its length in characters. The defaults are PHOC's defaults converted to
# edit units, so both methods start equally lenient: one edit costs ~9.2 PHOC L1 units on the query words and PHOC
# mass is 5 per character, hence temperature 2 / 9.2 ≈ 0.22 and background 0.5 * 5 / 9.2 ≈ 0.27 edits per character.
import os
from dataclasses import dataclass

import fargv
import torch

from dre import CharCodeEmbedder, JointSoftmaxObserver, LevenshteinDistance
from dre_run import run_dre


@dataclass
class Config:
    """dre with Levenshtein-distance emissions: one TREC run file for one collection version."""
    data_root: str = "data"
    collection: str = "trec5"
    version: str = "degrade5"
    output: str = ""
    temperature: float = 0.22
    "Softness of the emission around the background threshold, in edits."
    background: float = 0.27
    "Background distance, in edits per character of the token: a token at that distance from a query word is as likely that word as anything else."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    engine: str = "viterbi"
    "Matcher: viterbi (most probable match) or dfa (total probability of a match)."
    max_length: int = 32
    "Characters compared per token; longer tokens are truncated."
    workers: int = os.cpu_count() or 1
    "Accepted for a uniform Makefile interface; the distances are computed on the device instead."
    verbosity: int = fargv.FargvInt(0, short_name="v", is_count_switch=True)


def main() -> None:
    cfg, _ = fargv.parse(Config)

    def make_observer(vocabulary):
        return JointSoftmaxObserver(vocabulary, CharCodeEmbedder(cfg.max_length, device=cfg.device), LevenshteinDistance(),
                                    temperature=cfg.temperature, background=cfg.background, device=cfg.device)

    run_dre(cfg.data_root, cfg.collection, cfg.version, cfg.output, "levenshtein_soft", make_observer, cfg.device, verbosity=cfg.verbosity, engine=cfg.engine)


if __name__ == "__main__":
    main()
