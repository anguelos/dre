# PHOC → semantic embeddings

Trains `phoc.PHOCToSemantic` (`src/phoc/train_semantic.py`): a fixed PHOC followed by a small network predicting a
word's GloVe vector from its spelling, as a denoising task — inputs are corrupted with OCR-like edits, targets are the
clean words' vectors. It is compared with a learning-free baseline: look up the known word nearest in spelling
(Levenshtein or PHOC distance) and use its vector.

```bash
cd examples/phoc_semantic
make download            # GloVe 6B (~862 MB)
make train               # resumes from data/models/… if interrupted; EPOCHS=100 by default
make report              # runs the baselines on the same evaluation sets, prints the table
```

`MAX_WORDS` (default 100000, GloVe is frequency-ordered) and `DIM` (50/100/200/300) choose the data; `TRAIN_ARGS`
passes extra trainer flags, e.g. `make train TRAIN_ARGS=--hidden=1024,1024`.

## Evaluation

Words are split into training and held-out words. For each input the prediction counts as correct at top-1 / top-5 if
the true word is the nearest / among the five nearest of all words to the predicted vector (cosine):

| set | inputs | measures |
|---|---|---|
| `unseen_clean` | held-out words as spelled | generalisation to new words |
| `unseen_noisy` | held-out words, corrupted | new words through noise |
| `known_noisy` | training words in fresh corruptions | the OCR case: a known word, an unseen spelling |

The lookup baseline only knows training words, so it cannot recover unseen words by construction; on `known_noisy` it
is the bar the learned mapping has to clear.
