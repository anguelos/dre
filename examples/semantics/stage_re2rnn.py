#!/usr/bin/env python3
# Stage: download RE2RNN's data archive (datasets, rules, GloVe matrices; Jiang et al., EMNLP 2020) into
# <data_root>/raw/, verbatim, and record its SHA-256 in raw/SHA256SUMS (the Makefile's stamp for this step).
import hashlib
import os
import urllib.request
from pathlib import Path

import fargv

# The archive linked from github.com/jeffchy/RE2RNN (Google Drive file 1r8pMu8EwDjys8-6nHFiEJs4QtelhGqY8).
URL = "https://drive.usercontent.google.com/download?id=1r8pMu8EwDjys8-6nHFiEJs4QtelhGqY8&export=download&confirm=t"
SHA256 = "0de3617c848f18c4d021b721dca343f5ec786381dde55e63ef5a8f43daef715f"


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "force": False})
    raw = Path(p.data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "data.zip"
    if p.force or not archive.exists():
        part = raw / "data.zip.part"
        request = urllib.request.Request(URL, headers={"User-Agent": "dre-examples-stage/0.1 (+https://github.com/anguelos/dre)"})
        with urllib.request.urlopen(request) as response, open(part, "wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
        os.replace(part, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{archive} has SHA-256 {digest}, expected {SHA256}")
    (raw / "SHA256SUMS").write_text(f"{digest}  data.zip\n")


if __name__ == "__main__":
    main()
