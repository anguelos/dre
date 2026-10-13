![repo size](https://img.shields.io/github/repo-size/anguelos/dre)
![license](https://img.shields.io/github/license/anguelos/dre)

# dre

Probabilistic regular expressions over word sequences.

`dre` compiles a word-level regular expression into an automaton with one transition matrix per query word, and runs
it over documents whose tokens only *probably* are those words. How likely a token is to be a query word — its
emission — comes from any token-to-word distance: exact equality gives classic regex search, while PHOC embeddings,
edit distance or word embeddings let a pattern match through OCR errors, misspellings or synonyms. A document scores
the probability of its best match, so results can be ranked instead of merely filtered.

## Installation

```bash
pip install -e .
```

Text is tokenized with a naive built-in tokenizer unless NLTK is installed, in which case NLTK is used:

```bash
pip install -e .[nltk]
python -c 'import nltk; nltk.download("punkt")'
```

## Usage

A `Corpus` holds a collection of texts and answers word regexes over it:

```python
import dre

texts = ["the federal register of rules", "a register of federal rules", "the Federal tax register"]
corpus = dre.Corpus(texts)                      # exact word matching

corpus.scores("federal ( tax | . ) ? register")
# tensor([1., 0., 1.])                         one score per document, for ranking
corpus.search("federal ( tax | . ) ? register")
# [Hit(document=0, start=4, end=20, probability=1.0, text='federal register'),
#  Hit(document=2, start=4, end=24, probability=1.0, text='Federal tax register')]
```

It works in three layers, each with one job:
- **Corpus** (`search.py`): keeps the texts as one string, joined by an `END_OF_DOCUMENT` character; it maps what the
  engine answers back to documents and character spans, filters and ranks matches.
- **Tokenizer, observer** (`tokenizing.py`, `abstract_observer.py`): the tokenizer only splits text into word-like
  tokens (character spans, no normalization); an **observer** gives every distinct token the probability that it is
  each symbol of a regex, from an **embedder** (token → vector) and a **distance** between vectors.
- **Matcher** (`abstract_matcher.py`): runs the regex's automaton over the one token sequence and returns, for every
  token, the probability that a match ends there. `END_OF_DOCUMENT` returns the automaton to its initial state.

| stage | base class | implementations |
|---|---|---|
| embedder | `AbstractEmbedder` | `PHOCEmbedder`, `CharCodeEmbedder`, `TableEmbedder`, `Glove50Embedder`, `SymbolEmbedder` |
| distance | `AbstractDistance` (L1) | `L1Distance`, `CosineDistance`, `LevenshteinDistance` |
| observer | `AbstractObserver`, `SequenceObserver` (cached, exact) | `ExactObserver`, `JointSoftmaxObserver`, `SigmoidObserver` |
| matcher | `AbstractMatcher` (max-product) | `ViterbiMatcher` (windowed, batched), `DFAMatcher` (total probability) |

### Searching noisy text with PHOC

OCR breaks exact matching. With PHOC emissions a token resembling a query word is probably that word, and documents
are ranked by how well they match. A corpus takes the observer as a function of its vocabulary:

```python
noisy = ["the federa1 regisrer of rules",   # OCR errors
         "Federal Regster notices",         # a typo
         "a register of federal rules"]     # the words, but not the pattern

dre.Corpus(noisy).scores("federal register")
# tensor([0., 0., 0.])
soft = dre.Corpus(noisy, lambda vocabulary: dre.JointSoftmaxObserver(vocabulary, dre.PHOCEmbedder(), dre.L1Distance()))
soft.scores("federal register")
# tensor([0.989, 0.983, 0.000])
for hit in soft.search("federal register"):
    print(hit.document, hit.text, round(hit.probability, 3))
# 0 federa1 regisrer 0.989
# 1 Federal Regster 0.983
```

### Many queries over one collection

The observer embeds each distinct token once and caches its distance to every query word it has seen; `prepare`
computes the words of many queries in one pass:

```python
queries = ["federal register", "register of . rules", "notices"]
automata = [dre.compile_regex(q) for q in queries]
soft.observer.prepare(w for a in automata for w in dre.query_words(a))

for query, automaton in zip(queries, automata):
    print(query, soft.scores(automaton))
# federal register     tensor([0.989, 0.983, 0.000])
# register of . rules  tensor([0.000, 0.000, 0.922])
# notices              tensor([0.000, 1.000, 0.000])
```

### Any distance: edit distance

The joint softmax takes any distance and each token's scale ("mass") in the same units. For edit distance, tokens are
embedded as their character codes and the mass is the token's length:

```python
edits = dre.Corpus(noisy, lambda vocabulary: dre.JointSoftmaxObserver(vocabulary, dre.CharCodeEmbedder(), dre.LevenshteinDistance(),
                                                                      temperature=0.22, background=0.27))
edits.scores("federal register")
# tensor([0.978, 0.983, 0.000])
```

### Classifying sentences with rules

An anchored matcher requires a rule to cover the whole sentence, as rule-based classifiers do; token classes written
`<name>` match whatever tokens the given test accepts:

```python
classes = {"<number>": str.isdigit, "<punct>": lambda t: t in {".", ",", "!", "?"}}
sentences = dre.Corpus(["please call 0800 now !", "what time is it ?", "tell me what time it is ?"],
                       lambda vocabulary: dre.ExactObserver(vocabulary, classes=classes), dre.ViterbiMatcher(anchored=True))

sentences.scores(". * call <number> . *")
# tensor([1., 0., 0.])
sentences.scores("( what | who | when ) . * <punct>")
# tensor([0., 1., 0.])    — searching anywhere (not anchored) would also flag the third sentence
```

### The engine's answer, token by token

`Corpus.matches` returns what the matcher computes: for every token of the collection, the probability that a match
ends there, and where that match starts. `DFAMatcher` runs the determinized automaton: each token is read as exactly
one query word or none, so its probabilities are totals over all readings rather than the best path's:

```python
dfa = dre.Corpus(noisy, lambda vocabulary: dre.JointSoftmaxObserver(vocabulary, dre.PHOCEmbedder(), dre.L1Distance()),
                 dre.DFAMatcher())
matches = dfa.matches("federal register")
matches.end_probability   # one per token (rounded here); END_OF_DOCUMENT tokens close each document
# tensor([0.000, 0.000, 0.989, 0.000, 0.000, 0.000, 0.000, 0.983, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000])
matches.start             # where the match ending at each token starts (-1: below min_probability)
# tensor([-1, -1,  1, -1, -1, -1, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1])
```

`bin/drep` is a small grep built on this: `drep -v --regex="federal ( tax ) ? register" notes.txt`.

### PHOC embeddings

The `phoc` package is usable on its own: a learning-free, fixed-size embedding of a word's spelling, here with
counted (fractional) character occurrences per pyramid region and a thermometer-coded word length.

```python
from phoc import PHOC

phoc = PHOC(levels=(1, 2, 3, 5), max_len=20)
embeddings = phoc(["register", "regster", "registers", "banana"])   # (4, 416)
PHOC.distance(embeddings[:1], embeddings)
# tensor([[ 0.000,  9.333, 11.000, 58.000]])
```

## Query language

| syntax | matches |
|---|---|
| `word` | that word (case-insensitive; letters, `'` and `` ` ``) |
| `.` | any token |
| `<name>` | a token in a caller-defined class, e.g. `<number>` |
| `a b` | `a` followed by `b` |
| `a \| b` | `a` or `b` |
| `x ?`, `x *`, `x +` | optional, zero or more, one or more |
| `( … )` | grouping |

## Experiments

Self-contained, Makefile-driven experiments live in `examples/`; each downloads its own data:

```bash
cd examples/ir && make download compile baselines runs evals   # regex retrieval over OCR'd text (TREC-5 Confusion)
cd examples/semantics && make download compile report          # zero-shot rule classification (ATIS, QC, SMS)
```

## Development

```bash
make test       # unit, bugfixing and corner-case suites
make coverage   # coverage report over the unit suite
make doc        # HTML documentation in docs/_build/html
make build      # sdist + wheel
```

## Layout

- `src/dre/` — the engine: `wordre_parser.py` with `fa_segment.py` / `fa_graph.py` (regex → automaton),
  `abstract_embedder.py`, `abstract_distance.py`, `abstract_observer.py` (token → symbol probabilities),
  `abstract_matcher.py` (one token sequence in, a probability per token out), `search.py` (Corpus: texts in, ranked
  documents and character spans out)
- `src/phoc/` — PHOC embeddings, built step by step (`step01_interface.py`, `step02_overlap.py`)
- `examples/` — experiments (`ir/`, `semantics/`)
- `bin/` — standalone scripts
- `test/` — `unit_coverage`, `bugfixing`, `corner_cases`, `non_unit`
- `docs/` — Sphinx (MyST) documentation

## License

GPLv3
