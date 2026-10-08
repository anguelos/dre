#!/usr/bin/env python3
# Compile: extract one dimensionality of GloVe 6B from the archive → <data_root>/glove.6B.<dim>d.txt
from pathlib import Path
import zipfile

import fargv


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "dim": 100})
    name = f"glove.6B.{p.dim}d.txt"
    with zipfile.ZipFile(Path(p.data_root) / "raw" / "glove.6B.zip") as archive:
        archive.extract(name, p.data_root)


if __name__ == "__main__":
    main()
