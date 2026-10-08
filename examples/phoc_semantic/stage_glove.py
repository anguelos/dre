#!/usr/bin/env python3
# Stage: download GloVe 6B (Pennington et al., 2014; 400k words, 50/100/200/300-d) into <data_root>/raw/, verbatim,
# and record its SHA-256 in raw/SHA256SUMS, the Makefile's stamp for this step.
import hashlib
import os
import urllib.request
from pathlib import Path

import fargv
import tqdm

URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
SHA256 = "617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb"


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "force": False, "verbosity": fargv.FargvInt(0, short_name="v", is_count_switch=True)})
    raw = Path(p.data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "glove.6B.zip"
    if p.force or not archive.exists():
        part = raw / "glove.6B.zip.part"
        request = urllib.request.Request(URL, headers={"User-Agent": "dre-examples-stage/0.1 (+https://github.com/anguelos/dre)"})
        with urllib.request.urlopen(request) as response, open(part, "wb") as out:
            bar = tqdm.tqdm(total=int(response.headers.get("Content-Length", 0)) or None, unit="B", unit_scale=True) if p.verbosity >= 2 else None
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        os.replace(part, archive)
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise SystemExit(f"{archive} has SHA-256 {digest.hexdigest()}, expected {SHA256}")
    (raw / "SHA256SUMS").write_text(f"{digest.hexdigest()}  glove.6B.zip\n")


if __name__ == "__main__":
    main()
