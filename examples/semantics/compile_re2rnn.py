#!/usr/bin/env python3
# Compile: RE2RNN's archive → <data_root>/<DATASET>/ for ATIS, TREC (QC) and SMS:
#   labels.json                 label names in their index order (index 0 is where unmatched sentences go)
#   {dev,test}.clean.jsonl      {"label": ..., "tokens": [...]}, their exact tokenisation, BOS/EOS included
#   test.noise5.jsonl, test.noise20.jsonl   the test set with 5% / 20% of letters substituted (OCR-like noise)
#   rules.tsv                   label, original rule, rule in dre syntax
#   glove.npy, glove_vocab.txt  their 100-d GloVe rows and the token of each row
# Their files are pickles; they are loaded with an unpickler that refuses everything but NumPy array reconstruction.
import io
import json
import pickle
import random
import string
import zipfile
from pathlib import Path
from typing import List

import fargv
import numpy as np

from re2rnn_rules import RULE_FILES, read_rules, to_dre

NOISE = {"noise5": 0.05, "noise20": 0.20}


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        allowed = {("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct, ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")
        return allowed[(module, name)]


def degrade(tokens: List[str], probability: float, rng: random.Random) -> List[str]:
    # Substitutes letters of real tokens; BOS/EOS markers are left alone, digits and punctuation are not letters.
    return [token if token in ("BOS", "EOS") else "".join(rng.choice(string.ascii_lowercase) if c.isalpha() and rng.random() < probability else c for c in token)
            for token in tokens]


def write_jsonl(path: Path, labels: List[str], tokens: List[List[str]]) -> None:
    with open(path, "w") as f:
        for label, sentence in zip(labels, tokens):
            f.write(json.dumps({"label": label, "tokens": sentence}) + "\n")


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "seed": 1})
    root = Path(p.data_root)
    with zipfile.ZipFile(root / "raw" / "data.zip") as archive:
        for dataset, rule_file in RULE_FILES.items():
            out = root / dataset
            out.mkdir(parents=True, exist_ok=True)
            data = NumpyOnlyUnpickler(io.BytesIO(archive.read(f"data/{dataset}/dataset.pkl"))).load()
            i2t, i2in = data["i2t"], data["i2in"]
            labels = [i2in[n] for n in range(len(i2in))]
            (out / "labels.json").write_text(json.dumps(labels))
            for split in ("dev", "test"):
                sentences = [[i2t[int(t)] for t in query] for query in data[f"query_{split}"]]
                gold = [labels[int(np.asarray(intent).ravel()[0])] for intent in data[f"intent_{split}"]]
                write_jsonl(out / f"{split}.clean.jsonl", gold, sentences)
                if split == "test":
                    for version, probability in NOISE.items():
                        noisy = [degrade(s, probability, random.Random(f"{p.seed}-{dataset}-{version}-{n}")) for n, s in enumerate(sentences)]
                        write_jsonl(out / f"test.{version}.jsonl", gold, noisy)
            with open(out / "rules.tsv", "w") as f:
                for label, rule in read_rules(archive.read(f"data/{dataset}/{rule_file}").decode("utf-8")):
                    f.write(f"{label}\t{rule}\t{to_dre(rule)}\n")
            glove = NumpyOnlyUnpickler(io.BytesIO(archive.read(f"data/{dataset}/glove.100.emb"))).load()
            np.save(out / "glove.npy", glove.astype(np.float32))
            (out / "glove_vocab.txt").write_text("\n".join(i2t[n] for n in range(len(i2t))) + "\n")
    (root / ".compiled").write_text("ok\n")


if __name__ == "__main__":
    main()
