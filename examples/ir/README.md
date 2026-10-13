# IR experiments

Regex retrieval over OCR-degraded text. Everything is driven by the Makefile, run from this directory:

```bash
cd examples/ir
make help
make download            # TREC-5 Confusion track from NIST (~274 MB) into data/trec5/raw
make compile             # documents, tokens, queries, qrels
make run_re eval_re      # one method: runs for every version, then evaluation
make clean               # runs and evaluations only
make mrproper            # everything under data/, downloads included
```

`COLLECTION=synthetic` uses a small generated collection instead, for quick end-to-end checks.

## Protocol

- **Collection.** TREC-5 Confusion track: the 1994 Federal Register as `original` text and as OCR output at about
  5% (`degrade5`) and 20% (`degrade20`) character error rate, restricted to the 55,340 documents present in all three.
- **Queries.** Word-level regexes sampled from the original text by `compile_queries.py`: single words by
  document-frequency band, 2–3 word phrases, and patterns with `.`, `?` and `|` derived from 3-word phrases.
- **Relevance.** A document is relevant iff the query matches its original text (`compile_qrels.py`). Every
  document is tested, so the qrels are complete.
- **Runs.** Each method retrieves from one version and writes a TREC run file, `data/<collection>/runs/<method>/<version>.run`.

## Layout

| Prefix | Role |
|---|---|
| `stage_<collection>.py` | download (or generate) raw data, verbatim, with SHA-256 checksums |
| `compile_<collection>.py` | raw data → `docs/{original,degrade5,degrade20}.jsonl` |
| `compile_*.py` | collection-independent derived data: tokens, queries, qrels |
| `run_<method>.py` | one method on one version → TREC run file |
| `evaluate.py` | run + qrels → per-query scores and summary |

The baselines and the evaluation use optional packages: `pip install -e "../..[recommended]"` (BM25 also needs Java: `sudo apt install default-jre-headless`).
