#!/usr/bin/env python3
# Stage this example's external resources into <data_root>, checked by SHA-256: WordNet (unpacked under nltk_data/,
# where nltk.data finds it; augment.py's synonyms) and GloVe 6B (raw/glove.6B.zip, ~862 MB, or copied from
# --glove_zip), converted to glove/glove<dim>.pt (words in frequency order, float32 vectors) for each --dims.
# The RE2RNN and TREC-5 data come from examples/semantics and examples/ir (their own make targets).
import hashlib
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import fargv
import numpy as np
import torch

FILES = {  # target (relative to data_root) -> (url, sha256)
    "nltk_data/corpora/wordnet.zip": ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
                                      "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"),
    "raw/glove.6B.zip": ("https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip", "617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(root: Path, name: str, copy_from: str = "") -> Path:
    url, expected = FILES[name]
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_name(target.name + ".part")
        if copy_from:
            shutil.copyfile(copy_from, part)
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "dre-examples-stage/0.1 (+https://github.com/anguelos/dre)"})
            with urllib.request.urlopen(request) as response, open(part, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
        os.replace(part, target)
    actual = sha256(target)
    if actual != expected:
        raise SystemExit(f"{target} has SHA-256 {actual}, expected {expected}")
    return target


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "glove_zip": "", "dims": "100"})
    root = Path(p.data_root)
    with zipfile.ZipFile(fetch(root, "nltk_data/corpora/wordnet.zip")) as archive:
        archive.extractall(root / "nltk_data" / "corpora")
    glove_zip = None
    for dim in (int(d) for d in p.dims.split(",")):
        cache = root / "glove" / f"glove{dim}.pt"
        if cache.exists():
            continue
        glove_zip = glove_zip or fetch(root, "raw/glove.6B.zip", p.glove_zip)
        words, rows = [], []
        with zipfile.ZipFile(glove_zip) as archive, archive.open(f"glove.6B.{dim}d.txt") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                word, _, values = line.rstrip("\n").partition(" ")
                words.append(word)
                rows.append(np.array(values.split(" "), dtype=np.float32))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"words": words, "vectors": torch.from_numpy(np.stack(rows))}, cache)
        print(f"{cache}: {len(words)} words", flush=True)


if __name__ == "__main__":
    main()
