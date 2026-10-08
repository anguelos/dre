# Zero-shot rule-based classification (mirroring RE2RNN / FA-RNN)

Jiang et al. (EMNLP 2020) turn hand-written word-level regexes into FA-RNNs and evaluate them zero-shot on ATIS
(26 intents), QC/TREC-6 (6 question types) and SMS (spam). This folder runs the same rules, data and decision procedure
with dre, varying only how a sentence token is matched to a rule word (the emission model):

| method | a rule word matches… |
|---|---|
| `exact` | the identical token — the replication of their RE system |
| `glove_soft` | tokens with similar GloVe vectors (their own matrices) — semantic, like FA-RNN's GloVe relaxation |
| `levenshtein_soft` | tokens within a few edits — robust to misspellings |
| `phoc_soft` | tokens with similar PHOC embeddings |

Besides their clean test sets, `compile` writes OCR-like noisy versions (`noise5`, `noise20`: 5% / 20% of letters
substituted) to measure robustness.

```bash
cd examples/semantics
make download        # their archive (~18 MB) from the link in github.com/jeffchy/RE2RNN
make compile
make report          # runs and evaluates every method, prints the accuracy table
```

## Protocol, as in their code

- Rules (`[label]` headers, one rule per line) use `$` any token, `%` number, `&` punctuation; converted to dre syntax
  with `.`, `<number>`, `<punct>`, ranged quantifiers expanded (`re2rnn_rules.py`).
- A rule must match the whole sentence, BOS/EOS tokens included (`anchored=True`).
- A label scores the number of its matching rules, minus the matching rules of labels that take priority over it
  (`PRIORITIES`, from their `create_logic_mat_bias.py`); the arg-max wins, ties and no match going to label index 0.
- Their files are pickles; they are loaded with an unpickler that accepts nothing but NumPy array reconstruction.

## Replication status

`exact` gives 85.78 / 60.40 / 93.20 (ATIS / QC / SMS) against their published 87.01 / 64.40 / 93.20. The automata
shipped in their archive give exactly our numbers on ATIS and SMS under the same semantics, and lower on QC; the
released rule files, the archived automata and the paper do not fully agree with each other, so the remaining gap
could not be traced. All dre methods share the same rules and protocol, so they are compared with each other on equal
terms.
