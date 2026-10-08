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

The examples share one helper that turns texts into a vocabulary and token-index tensors:

```python
import dre

def encode(documents):
    tokens = [d.lower().split() for d in documents]
    vocabulary = dre.Vocabulary.from_documents(tokens)
    return vocabulary, [vocabulary.encode(d) for d in tokens]
```

The engine has three interchangeable stages:
- an **observer** gives, for every vocabulary token, the probability that it is each symbol of a regex; soft
  observers derive it from an **embedder** (token → vector) and a **distance** between vectors;
- a **matcher** runs the regex's automaton over documents of token ids and scores each document by its best match.

| stage | base class | implementations |
|---|---|---|
| embedder | `AbstractEmbedder` | `PHOCEmbedder`, `CharCodeEmbedder`, `TableEmbedder`, `Glove50Embedder`, `SymbolEmbedder` |
| distance | `AbstractDistance` (L1) | `L1Distance`, `CosineDistance`, `LevenshteinDistance` |
| observer | `AbstractObserver`, `SequenceObserver` (cached, exact) | `ExactObserver`, `JointSoftmaxObserver`, `SigmoidObserver` |
| matcher | `AbstractMatcher` (max-product) | `ViterbiMatcher` (windowed, batched), `DFAMatcher` (total probability) |

### Classic regex search

With exact emissions a token either is a query word or is not, and dre behaves like a regex engine over words:

```python
vocabulary, documents = encode(["the federal register of rules",
                                "a register of federal rules",
                                "the federal tax register"])

automaton = dre.compile_regex("federal ( tax | . ) ? register")
observer = dre.ExactObserver(vocabulary)
matcher = dre.ViterbiMatcher()
matcher.scores(automaton, observer.observe(automaton), documents)
# tensor([1., 0., 1.])
```

### Searching noisy text with PHOC

OCR breaks exact matching. With PHOC emissions a token resembling a query word is probably that word, and documents
are ranked by how well they match:

```python
from phoc import PHOC

vocabulary, documents = encode(["the federa1 regisrer of rules",   # OCR errors
                                "Federal Regster notices",         # a typo
                                "a register of federal rules"])    # the words, but not the pattern

automaton = dre.compile_regex("federal register")
exact = dre.ExactObserver(vocabulary)
soft = dre.JointSoftmaxObserver(vocabulary, dre.PHOCEmbedder(PHOC()), dre.L1Distance())
matcher.scores(automaton, exact.observe(automaton), documents)
# tensor([0., 0., 0.])
matcher.scores(automaton, soft.observe(automaton), documents)
# tensor([0.989, 0.983, 0.000])
```

### Many queries over one collection

An observer serves any number of regexes. It embeds the vocabulary per distinct token, not per running token, and
caches the distances to every query word it has seen; `prepare` computes the words of many queries in one pass:

```python
queries = ["federal register", "register of . rules", "notices"]
automata = [dre.compile_regex(q) for q in queries]
soft.prepare(w for a in automata for w in dre.query_words(a))

for query, automaton in zip(queries, automata):
    print(query, matcher.scores(automaton, soft.observe(automaton), documents))
# federal register     tensor([0.989, 0.983, 0.000])
# register of . rules  tensor([0.000, 0.000, 0.922])
# notices              tensor([0.000, 1.000, 0.000])
```

### Any distance: edit distance

The joint softmax takes any distance and each token's scale ("mass") in the same units. For edit distance, tokens are
embedded as their character codes and the mass is the token's length:

```python
edits = dre.JointSoftmaxObserver(vocabulary, dre.CharCodeEmbedder(), dre.LevenshteinDistance(),
                                 temperature=0.22, background=0.27)
matcher.scores(automaton, edits.observe(automaton), documents)
# tensor([0.978, 0.983, 0.000])
```

### Classifying sentences with rules

`anchored=True` requires a rule to cover the whole sentence, as rule-based classifiers do; token classes written
`<name>` match whatever tokens the given test accepts:

```python
classes = {"<number>": str.isdigit, "<punct>": lambda t: t in {".", ",", "!", "?"}}
rules = {"spam": dre.compile_regex(". * call <number> . *"),
         "question": dre.compile_regex("( what | who | when ) . * <punct>")}
vocabulary, sentences = encode(["please call 0800 now !",
                                "what time is it ?",
                                "tell me what time it is ?"])
observer = dre.ExactObserver(vocabulary, classes=classes)
anchored = dre.ViterbiMatcher(anchored=True)

for label, automaton in rules.items():
    print(label, anchored.scores(automaton, observer.observe(automaton), sentences))
# spam      tensor([1., 0., 0.])
# question  tensor([0., 1., 0.])    — searching anywhere (anchored=False) would also flag the third sentence
```

### Searching texts

A `Corpus` tokenizes raw texts once (tensor operations over pylelemmatize character codes; tokens keep their
character spans) and a `Searcher` returns the matches over the whole collection, most probable first, as character
spans. `DFAMatcher` runs the determinized automaton: each token is read as exactly one query word or none, the forward
pass keeps a distribution over DFA states, and a match's probability is the total over all readings rather than the
best path's.

```python
texts = ["The Federa1 Regisrer, vol. 3: federal rules.",
         "Notices of the Federal Register.",
         "A register of federal rules."]
corpus = dre.Corpus(texts)
observer = dre.JointSoftmaxObserver(corpus.vocabulary, dre.PHOCEmbedder(PHOC()), dre.L1Distance())
searcher = dre.Searcher(corpus, observer, dre.DFAMatcher())

for hit in searcher.search("federal ( register | rules )"):
    print(hit.document, texts[hit.document][hit.start:hit.end], round(hit.probability, 3))
# 1 Federal Register 1.0
# 0 federal rules 0.998
# 2 federal rules 0.998
# 0 Federa1 Regisrer 0.989
```

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
  `abstract_matcher.py` (scoring documents, finding matches), `search.py` (texts in, ranked character spans out)
- `src/phoc/` — PHOC embeddings, built step by step (`step01_interface.py`, `step02_overlap.py`)
- `examples/` — experiments (`ir/`, `semantics/`)
- `bin/` — standalone scripts
- `test/` — `unit_coverage`, `bugfixing`, `corner_cases`, `non_unit`
- `docs/` — Sphinx (MyST) documentation

## License

GPLv3
