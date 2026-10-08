import gzip
import json
import multiprocessing
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, TextIO, Tuple, TypeVar, Union

from dre import Corpus, Tokenizer

PathLike = Union[str, Path]
Run = Dict[str, List[Tuple[str, float]]]
Qrels = Dict[str, Dict[str, int]]


T = TypeVar("T")
R = TypeVar("R")


def parallel_imap(function: Callable[[T], R], items: Iterable[T], workers: int, chunksize: int = 64) -> Iterator[R]:
    # Results come back in input order, so outputs stay deterministic whatever the number of workers.
    if workers <= 1:
        yield from map(function, items)
        return
    with multiprocessing.Pool(workers) as pool:
        yield from pool.imap(function, items, chunksize=chunksize)


class Query(NamedTuple):
    qid: str
    family: str
    regex: str


def collection_dir(data_root: PathLike, collection: str) -> Path:
    return Path(data_root) / collection


@contextmanager
def atomic_write(path: PathLike) -> Iterator[TextIO]:
    # Make treats an existing file as up to date, so a partially written output must never appear under its final name.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8") as f:
        yield f
    os.replace(part, path)


def open_text(path: PathLike) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_trec_sgml(text: str) -> Iterator[Tuple[str, str]]:
    for doc in re.finditer(r"<DOC>(.*?)</DOC>", text, flags=re.S):
        body = doc.group(1)
        docno = re.search(r"<DOCNO>\s*(.*?)\s*</DOCNO>", body, flags=re.S)
        if docno is None:
            raise ValueError(f"<DOC> without <DOCNO>: {body[:80]!r}")
        content = re.sub(r"<DOCNO>.*?</DOCNO>", " ", body, flags=re.S)
        content = re.sub(r"<[^>]*>", " ", content)
        yield docno.group(1), content


def write_docs(path: PathLike, docs: Iterable[Tuple[str, str]]) -> int:
    count = 0
    with atomic_write(path) as f:
        for docno, text in docs:
            f.write(json.dumps({"docno": docno, "text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_docs(path: PathLike) -> Iterator[Tuple[str, str]]:
    with open_text(path) as f:
        for line in f:
            record = json.loads(line)
            yield record["docno"], record["text"]


def write_token_strings(path: PathLike, token_strings: Iterable[Tuple[str, str]]) -> int:
    count = 0
    with atomic_write(path) as f:
        for docno, token_string in token_strings:
            f.write(f"{docno}\t{token_string}\n")
            count += 1
    return count


def read_token_strings(path: PathLike) -> Iterator[Tuple[str, str]]:
    with open_text(path) as f:
        for line in f:
            docno, token_string = line.rstrip("\n").split("\t", 1)
            yield docno, token_string


def read_corpus(path: PathLike) -> Tuple[List[str], Corpus]:
    # A tokens file as dre consumes it: docnos, and the token strings as a Corpus (tokens are already split and
    # lower-cased, so whitespace is the only separator).
    documents = list(read_token_strings(path))
    return [docno for docno, _ in documents], Corpus([tokens for _, tokens in documents], Tokenizer.pretokenized(), lowercase=False)

def write_queries(path: PathLike, queries: Iterable[Query]) -> None:
    with atomic_write(path) as f:
        for query in queries:
            f.write(f"{query.qid}\t{query.family}\t{query.regex}\n")


def read_queries(path: PathLike) -> List[Query]:
    with open_text(path) as f:
        return [Query(*line.rstrip("\n").split("\t", 2)) for line in f if line.strip()]


def write_qrels(path: PathLike, qrels: Qrels) -> None:
    with atomic_write(path) as f:
        for qid in sorted(qrels):
            for docno in sorted(qrels[qid]):
                f.write(f"{qid} 0 {docno} {qrels[qid][docno]}\n")


def read_qrels(path: PathLike) -> Qrels:
    qrels: Qrels = {}
    with open_text(path) as f:
        for line in f:
            if line.strip():
                qid, _, docno, rel = line.split()
                qrels.setdefault(qid, {})[docno] = int(rel)
    return qrels


def write_run(path: PathLike, run: Run, tag: str, depth: int = 1000) -> None:
    # trec_eval re-sorts by score and breaks ties by docno in reverse order; sorting ties by docno here and writing
    # the same order makes the file self-consistent, and the tie policy explicit.
    with atomic_write(path) as f:
        for qid in sorted(run):
            ranked = sorted(run[qid], key=lambda item: (-item[1], item[0]))[:depth]
            for rank, (docno, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docno} {rank} {score:.6g} {tag}\n")


def read_run(path: PathLike) -> Run:
    run: Run = {}
    with open_text(path) as f:
        for line in f:
            if line.strip():
                qid, _, docno, _, score, _ = line.split()
                run.setdefault(qid, []).append((docno, float(score)))
    return run
