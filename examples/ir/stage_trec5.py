#!/usr/bin/env python3
# Stage: download the TREC-5 Confusion track from NIST into <data_root>/trec5/raw/, verbatim.
# SHA256SUMS is written last; it records exactly which files were used and is the Makefile's stamp for this step.
import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Dict

import fargv
import tqdm

from trec_io import atomic_write

NIST = "https://trec.nist.gov/data"
# NIST's CDN rejects urllib's default user agent; identify the script honestly instead.
USER_AGENT = "dre-examples-stage/0.1 (+https://github.com/anguelos/dre)"
FILES = {
    "confusion_track.tar.gz": f"{NIST}/confusion/confusion_track.tar.gz",
    "README": f"{NIST}/confusion/README",
    "topics.confusion.cf1-cf50.gz": f"{NIST}/topics_eng/topics.confusion.cf1-cf50.gz",
    "confusion.known_items.gz": f"{NIST}/qrels_eng/confusion.known_items.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sums(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return {name: digest for digest, name in (line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())}


def download(url: str, dest: Path, verbosity: int) -> None:
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length", 0)) or None
        bar = tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if verbosity >= 2 else None
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()
    os.replace(part, dest)


def main() -> None:
    p, _ = fargv.parse({
        "data_root": "data",
        "force": False,
        "verbosity": fargv.FargvInt(0, short_name="v", is_count_switch=True),
    })
    raw = Path(p.data_root) / "trec5" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sums_path = raw / "SHA256SUMS"
    recorded = read_sums(sums_path)
    sums = {}
    for name, url in FILES.items():
        dest = raw / name
        if dest.exists() and not p.force:
            if p.verbosity >= 1:
                print(f"present: {dest}")
        else:
            if p.verbosity >= 1:
                print(f"downloading {url}")
            download(url, dest, p.verbosity)
        sums[name] = sha256(dest)
        if name in recorded and recorded[name] != sums[name] and not p.force:
            raise SystemExit(f"Checksum of {dest} differs from {sums_path}; rerun with --force to accept the new file")
    with atomic_write(sums_path) as f:
        for name in sorted(sums):
            f.write(f"{sums[name]}  {name}\n")


if __name__ == "__main__":
    main()
