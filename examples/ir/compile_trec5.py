#!/usr/bin/env python3
# Compile: TREC-5 Confusion tarball → <data_root>/trec5/docs/{original,degrade5,degrade20}.jsonl
# Reads the tarball in place, in one sequential pass (a .tar.gz cannot be seeked cheaply), then parses its daily files
# in parallel. Markup and comments are stripped, entities decoded, whitespace collapsed, and all three versions are
# restricted to the same documents: those present in every version and not listed in omitted_docs.
import gzip
import html
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import fargv

from trec_io import iter_trec_sgml, parallel_imap, read_docs, write_docs

VERSIONS = ("original", "degrade5", "degrade20")
# Federal Register entities html.unescape does not know; anything else unknown becomes a space.
FR_ENTITIES = {"&hyph;": "-", "&blank;": " ", "&cir;": "o", "&sect;": "§"}


def clean(text: str) -> str:
    text = re.sub(r"&[A-Za-z0-9#]+;", lambda m: FR_ENTITIES.get(m.group(0), m.group(0)), text)
    text = html.unescape(text)
    text = re.sub(r"&[A-Za-z0-9#]+;", " ", text)
    return " ".join(text.split())


def parse_member(compressed: bytes) -> List[Tuple[str, str]]:
    text = gzip.decompress(compressed).decode("utf-8", errors="replace")
    # <PARENT> holds the parent document's DOCNO: metadata that was never printed, so absent from the OCR versions.
    text = re.sub(r"<PARENT>.*?</PARENT>", " ", text, flags=re.S)
    return [(docno, clean(content)) for docno, content in iter_trec_sgml(text)]


def main() -> None:
    p, _ = fargv.parse({
        "data_root": "data",
        "workers": os.cpu_count() or 1,
        "verbosity": fargv.FargvInt(0, short_name="v", is_count_switch=True),
    })
    root = Path(p.data_root) / "trec5"
    docs_dir = root / "docs"
    members: Dict[str, Dict[str, bytes]] = {version: {} for version in VERSIONS}
    omitted: Set[str] = set()
    with tarfile.open(root / "raw" / "confusion_track.tar.gz", "r|gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if not member.isfile():
                continue
            if member.name == "confusion_track/omitted_docs":
                omitted = set(tar.extractfile(member).read().decode().split())
            elif len(parts) > 2 and parts[1] in VERSIONS:
                members[parts[1]][member.name] = tar.extractfile(member).read()
    docnos: Dict[str, Set[str]] = {}
    for version in VERSIONS:
        names = sorted(members[version], key=str.lower)
        seen: Set[str] = set()

        def unique_docs(names: List[str] = names, version: str = version, seen: Set[str] = seen) -> Iterator[Tuple[str, str]]:
            for parsed in parallel_imap(parse_member, (members[version][name] for name in names), p.workers, chunksize=1):
                for docno, text in parsed:
                    if docno in seen:
                        raise ValueError(f"Duplicate {docno} in {version}")
                    seen.add(docno)
                    yield docno, text

        write_docs(docs_dir / f"{version}.all.jsonl", unique_docs())
        docnos[version] = seen
    kept = set.intersection(*docnos.values()) - omitted
    for version in VERSIONS:
        everything = docs_dir / f"{version}.all.jsonl"
        count = write_docs(docs_dir / f"{version}.jsonl", ((d, t) for d, t in read_docs(everything) if d in kept))
        os.remove(everything)
        if p.verbosity >= 1:
            print(f"{version}: {len(docnos[version])} documents, {count} kept")
    with open(docs_dir / "stats.json", "w") as f:
        json.dump({"kept": len(kept), "omitted_listed": len(omitted), **{v: len(d) for v, d in docnos.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
