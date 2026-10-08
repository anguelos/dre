#!/usr/bin/env python3
# Stage: download the out-of-vocabulary evaluation material into <data_root>/raw/, verbatim:
#   rw.zip          Stanford Rare Words similarity pairs (Luong et al., 2013)
#   card-660.tsv    Cambridge Rare Word Dataset similarity pairs (Pilehvar et al., EMNLP 2018)
#   wiki-news-300d-1M-subword.bin.zip   fastText English vectors with subword information (Mikolov et al., 2018),
#                   which can build a vector for any string: the standard out-of-vocabulary baseline (~4.5 GB)
# Checksums are pinned; raw/EVAL_SHA256SUMS is written last and is the Makefile's stamp for this step.
import hashlib
import os
import urllib.request
from pathlib import Path

import fargv
import tqdm

FILES = {
    "rw.zip": ("https://nlp.stanford.edu/~lmthang/morphoNLM/rw.zip", "057ca4b47ea9515bd90c9147ba4d6b8aa6a4df3732e3266bc731144743f5508d"),
    "card-660.tsv": ("https://pilehvar.github.io/card-660/dataset.tsv", "0cfae4d9ee596c4dd745a138f49c99e74c31a8e171bc4108b62d243607a6c1a8"),
    "wiki-news-300d-1M-subword.bin.zip": ("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip", "a4d513b270b43674426ea4b16685ca63f26b651ce9156e33f1283818cbe8db84"),
}
USER_AGENT = "dre-examples-stage/0.1 (+https://github.com/anguelos/dre)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "verbosity": fargv.FargvInt(0, short_name="v", is_count_switch=True)})
    raw = Path(p.data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sums = []
    for name, (url, expected) in FILES.items():
        path = raw / name
        if not path.exists():
            part = raw / (name + ".part")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response, open(part, "wb") as out:
                total = int(response.headers.get("Content-Length", 0)) or None
                bar = tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=name) if p.verbosity >= 2 else None
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    out.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            os.replace(part, path)
        digest = sha256(path)
        if expected and digest != expected:
            raise SystemExit(f"{path} has SHA-256 {digest}, expected {expected}")
        sums.append(f"{digest}  {name}")
    (raw / "EVAL_SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
