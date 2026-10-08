#!/usr/bin/env python3
# Task B — out-of-vocabulary (OOV) embedding imputation, on the rare-word similarity benchmarks RW and CARD-660.
# The "known vocabulary" is the trained models' training words; a known word keeps its GloVe vector and every other
# word gets a vector from the method under test (the protocol of Mimick and LOVE). The score is the Spearman
# correlation between human similarity ratings and the cosine similarity of the two vectors; a word a method cannot
# give a vector to counts as similarity 0, so coverage gaps are not hidden by dropping pairs. Each benchmark is also
# evaluated with OCR-like noise on its words (version "noisy"), where most words fall out of the known vocabulary.
# A phrase ("formic_acid") gets the mean vector of its words. 95% confidence intervals come from a paired bootstrap
# over pairs (the same resamples for every method), which also gives, for every learned model and every other method,
# the fraction of resamples in which the learned model scores higher.
import glob
import json
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fargv
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from phoc import PHOC, PHOCToSemantic
from phoc.train_semantic import Config as TrainConfig
from phoc.train_semantic import corrupt, load_vectors

try:  # optional: only the fastText rows need it (requirements.txt); they are skipped without it
    import fasttext
except ImportError:
    fasttext = None

Pairs = List[Tuple[str, str, float]]
Vectors = Dict[str, Optional[np.ndarray]]
Method = Callable[[List[str]], Vectors]  # all words a method must embed at once → word: vector (None: no vector)


@dataclass
class Config:
    """Evaluate OOV vector imputation on RW and CARD-660, clean and with OCR-like noise."""
    data_root: str = "data"
    states: str = "data/models/*.pt"
    "Trained PHOCToSemantic state files (glob); the first one's split defines the known vocabulary."
    glove: str = "data/glove.6B.100d.txt"
    "Full GloVe file, for the oracle row (every word GloVe has, no imputation)."
    fasttext: str = "data/wiki-news-300d-1M-subword.bin"
    "fastText subword model; the fastText row is skipped if it or the fasttext package is missing."
    noise: float = 0.1
    seed: int = 1
    bootstrap: int = 1000
    output: str = "data/results/imputation.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def read_benchmarks(raw: Path) -> Dict[str, Pairs]:
    with zipfile.ZipFile(raw / "rw.zip") as archive:
        rw = [line.split("\t") for line in archive.read("rw/rw.txt").decode("utf-8").splitlines() if line.strip()]
    card = [line.split("\t") for line in (raw / "card-660.tsv").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"RW": [(a, b, float(s)) for a, b, s, *_ in rw], "CARD-660": [(a, b, float(s)) for a, b, s in card]}


def words_of(phrase: str) -> List[str]:
    return [w for w in re.split(r"[_\s]+", phrase.lower()) if w]


def ranks(values: np.ndarray) -> np.ndarray:
    # Average ranks, ties sharing the mean of their positions.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def similarities(pairs: Pairs, vectors: Vectors) -> Tuple[np.ndarray, float]:
    # Cosine similarity of every pair (0 if a side has no vector) and the fraction of pairs without vectors.
    def phrase_vector(phrase: str) -> Optional[np.ndarray]:
        found = [v for v in (vectors.get(w) for w in words_of(phrase)) if v is not None]
        return np.mean(found, axis=0) if found else None

    out, missing = [], 0
    for a, b, _ in pairs:
        va, vb = phrase_vector(a), phrase_vector(b)
        if va is None or vb is None:
            missing += 1
            out.append(0.0)
        else:
            out.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)))
    return np.array(out), missing / len(pairs)


def main() -> None:
    cfg, _ = fargv.parse(Config)
    benchmarks = read_benchmarks(Path(cfg.data_root) / "raw")
    versions: Dict[str, Pairs] = {}
    for name, pairs in benchmarks.items():
        versions[f"{name} clean"] = pairs
        versions[f"{name} noisy"] = [(" ".join(corrupt(w, cfg.noise, random.Random(f"{cfg.seed}-{w}")) for w in words_of(a)),
                                      " ".join(corrupt(w, cfg.noise, random.Random(f"{cfg.seed}-{w}")) for w in words_of(b)), s) for a, b, s in pairs]
    all_words = sorted({w for pairs in versions.values() for a, b, _ in pairs for p in (a, b) for w in words_of(p)})
    device = torch.device(cfg.device)

    glove: Dict[str, np.ndarray] = {}
    with open(cfg.glove, encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            glove[word] = np.asarray(values, dtype=np.float32)

    state_paths = sorted(glob.glob(cfg.states))
    first = torch.load(state_paths[0], map_location="cpu", weights_only=False)  # our own file
    train_cfg = TrainConfig(**{k: v for k, v in first["config"].items() if k in TrainConfig.__dataclass_fields__})
    words, vectors = load_vectors(train_cfg)
    known_words = first["split"]["train"]
    known_set = set(known_words)
    known = {w: vectors[n] for n, w in enumerate(words) if w in known_set}
    unknown = [w for w in all_words if w not in known_set]

    def lookup_levenshtein(_: List[str]) -> Vectors:
        nearest = np.concatenate([cdist(unknown[b:b + 512], known_words, scorer=Levenshtein.distance, dtype=np.int32, workers=-1).argmin(axis=1)
                                  for b in range(0, len(unknown), 512)])
        return {**known, **{w: known[known_words[n]] for w, n in zip(unknown, nearest)}}

    def lookup_phoc(_: List[str]) -> Vectors:
        phoc = PHOC().to(device)
        reference = torch.cat([phoc(known_words[b:b + 8192]) for b in range(0, len(known_words), 8192)])
        nearest = torch.cat([torch.cdist(phoc(unknown[b:b + 512]), reference, p=1).argmin(dim=1).cpu() for b in range(0, len(unknown), 512)])
        return {**known, **{w: known[known_words[int(n)]] for w, n in zip(unknown, nearest)}}

    def learned(path: str, impute_known: bool) -> Method:
        def method(_: List[str]) -> Vectors:
            model = PHOCToSemantic.from_checkpoint(path).to(device)
            targets = all_words if impute_known else unknown
            with torch.no_grad():
                predicted = torch.cat([model(targets[b:b + 1024]).cpu() for b in range(0, len(targets), 1024)]).numpy()
            return {**({} if impute_known else known), **dict(zip(targets, predicted))}
        return method

    methods: Dict[str, Method] = {
        "GloVe, full 400k vocabulary (oracle, no imputation)": lambda ws: {w: glove.get(w) for w in ws},
        "known words only (OOV: no vector)": lambda ws: {w: known.get(w) for w in ws},
        "lookup: nearest known spelling, Levenshtein": lookup_levenshtein,
        "lookup: nearest known spelling, PHOC": lookup_phoc,
    }
    learned_names = []
    for path in state_paths:
        name = Path(path).stem.split(".")[-1]
        methods[f"learned {name}: OOV imputed"] = learned(path, impute_known=False)
        methods[f"learned {name}: every word imputed"] = learned(path, impute_known=True)
        learned_names += [f"learned {name}: OOV imputed", f"learned {name}: every word imputed"]
    if fasttext is not None and Path(cfg.fasttext).exists():
        fasttext_model = fasttext.load_model(cfg.fasttext)
        methods["fastText subword (its own 300-d space; every word)"] = lambda ws: {w: fasttext_model.get_word_vector(w) for w in ws}

    resamples = {v: np.random.default_rng(cfg.seed).integers(0, len(pairs), size=(cfg.bootstrap, len(pairs))) for v, pairs in versions.items()}
    gold = {v: np.array([s for *_, s in pairs]) for v, pairs in versions.items()}
    results: Dict = {"oov_rate": {v: float(np.mean([all(w not in known_set for w in words_of(p)) for a, b, _ in pairs for p in (a, b)]))
                                  for v, pairs in versions.items()}, "scores": {}, "learned_beats": {}}
    boot: Dict[str, Dict[str, np.ndarray]] = {}
    for method, embed in methods.items():
        vectors_of = embed(all_words)
        results["scores"][method], boot[method] = {}, {}
        for version, pairs in versions.items():
            sims, missing = similarities(pairs, vectors_of)
            boot[method][version] = np.array([spearman(gold[version][r], sims[r]) for r in resamples[version]])
            low, high = np.percentile(boot[method][version], [2.5, 97.5])
            results["scores"][method][version] = {"spearman": spearman(gold[version], sims), "ci95": [float(low), float(high)], "pairs_without_vector": missing}
        print(method, {v: round(s["spearman"], 3) for v, s in results["scores"][method].items()}, flush=True)
    for name in learned_names:
        results["learned_beats"][name] = {other: {v: float(np.mean(boot[name][v] > boot[other][v])) for v in versions}
                                         for other in methods if other != name}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
