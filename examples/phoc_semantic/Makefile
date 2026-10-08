# PHOC → semantic embeddings (src/phoc/train_semantic.py) trained on GloVe 6B, against nearest-known-word lookups.
# Run from examples/phoc_semantic; `make help` lists the targets.
PYTHON ?= python3
DATA ?= data
DIM ?= 100
MAX_WORDS ?= 100000
EPOCHS ?= 100
NOISE ?= 0.1
# Evaluated once (:=); a recursive variable would import torch again wherever DEVICE is used.
DEVICE := $(or $(DEVICE),$(shell $(PYTHON) -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"))
TRAIN_ARGS ?=
# Each variant (different TRAIN_ARGS) needs its own NAME: a state file only resumes with the settings it was made with.
NAME ?= default
# Decoder variants (task A), likewise named.
DECODER ?= default
DECODER_ARGS ?=
DECODER_EPOCHS ?= 100
export PYTHONPATH := ../../src$(if $(PYTHONPATH),:$(PYTHONPATH))

VECTORS := $(DATA)/glove.6B.$(DIM)d.txt
STATE := $(DATA)/models/phoc_semantic.$(DIM)d.$(MAX_WORDS).$(NAME).pt
RESULTS := $(DATA)/results/$(DIM)d.$(MAX_WORDS)
DISTANCES := levenshtein phoc
FASTTEXT := $(DATA)/wiki-news-300d-1M-subword.bin
DECODER_STATE := $(DATA)/models/phoc_decoder.$(DIM)d.$(MAX_WORDS).$(DECODER).pt

.DELETE_ON_ERROR:
.PHONY: help download download_eval compile train train_decoder baselines report imputation correction clean mrproper

help:
	@echo "make download     GloVe 6B (~862 MB) into $(DATA)/raw; never triggered implicitly"
	@echo "make compile      extract the $(DIM)-d vectors"
	@echo "make train        train (or resume) $(STATE) up to EPOCHS=$(EPOCHS); MAX_WORDS=$(MAX_WORDS) most frequent words"
	@echo "make baselines    nearest-known-word lookup by: $(DISTANCES), on the trained model's evaluation sets"
	@echo "make report       every trained variant (NAME) against the baselines"
	@echo "make download_eval  RW, CARD-660 and fastText (~4.5 GB) for task B; never triggered implicitly"
	@echo "make imputation   task B: OOV vector imputation on RW and CARD-660, clean and noisy, every variant vs baselines"
	@echo "make train_decoder  task A: train (or resume) the PHOC -> spelling decoder DECODER=$(DECODER) up to DECODER_EPOCHS=$(DECODER_EPOCHS)"
	@echo "make correction   task A: spelling correction, every decoder vs identity, lookups and a noisy channel"
	@echo "make clean        remove models and results (downloads and extracted vectors are kept)"
	@echo "make mrproper     remove everything under $(DATA)"
	@echo "variables: DIM MAX_WORDS EPOCHS NOISE DEVICE NAME TRAIN_ARGS (extra trainer flags, e.g. --hidden=1024,1024)"

download:
	$(PYTHON) stage_glove.py --data_root=$(DATA) -vv

$(DATA)/raw/SHA256SUMS:
	@echo "$@ is missing: run 'make download' first" >&2; exit 1

$(VECTORS): $(DATA)/raw/SHA256SUMS compile_glove.py
	$(PYTHON) compile_glove.py --data_root=$(DATA) --dim=$(DIM)

compile: $(VECTORS)

# Always invoked: the trainer resumes from STATE, or does nothing once EPOCHS are done.
train: $(VECTORS)
	@mkdir -p $(DATA)/models
	$(PYTHON) -m phoc.train_semantic --vectors=$(VECTORS) --output=$(STATE) --epochs=$(EPOCHS) --max_words=$(MAX_WORDS) --noise=$(NOISE) --device=$(DEVICE) $(TRAIN_ARGS)

# The baselines read the split from STATE, so they need a trained (or at least started) model; only its existence
# matters (order-only), as they do not depend on the weights that each epoch rewrites.
$(RESULTS)/baseline_%.json: baseline_lookup.py | $(STATE)
	$(PYTHON) baseline_lookup.py --state=$(STATE) --distance=$* --output=$@ --device=$(DEVICE)

$(STATE):
	@echo "$@ is missing: run 'make train' first" >&2; exit 1

baselines: $(foreach d,$(DISTANCES),$(RESULTS)/baseline_$(d).json)

$(DATA)/raw/EVAL_SHA256SUMS:
	@echo "$@ is missing: run 'make download_eval' first" >&2; exit 1

download_eval:
	$(PYTHON) stage_eval.py --data_root=$(DATA) -vv

$(FASTTEXT): $(DATA)/raw/EVAL_SHA256SUMS
	$(PYTHON) -c "import zipfile; zipfile.ZipFile('$(DATA)/raw/wiki-news-300d-1M-subword.bin.zip').extract('$(notdir $(FASTTEXT))', '$(DATA)')"

# Always re-run: it scores whatever state the trained variants are in now.
imputation: $(VECTORS) $(FASTTEXT)
	$(PYTHON) evaluate_imputation.py --data_root=$(DATA) --states="$(DATA)/models/phoc_semantic.$(DIM)d.$(MAX_WORDS).*.pt" --glove=$(VECTORS) --fasttext=$(FASTTEXT) --output=$(RESULTS)/imputation.json --device=$(DEVICE)

# Always invoked: the trainer resumes from DECODER_STATE, or does nothing once DECODER_EPOCHS are done.
train_decoder: $(VECTORS)
	@mkdir -p $(DATA)/models
	$(PYTHON) -m phoc.train_decoder --vectors=$(VECTORS) --output=$(DECODER_STATE) --epochs=$(DECODER_EPOCHS) --max_words=$(MAX_WORDS) --noise=$(NOISE) --device=$(DEVICE) $(DECODER_ARGS)

# Always re-run: it scores whatever state the trained decoders are in now.
correction: $(VECTORS)
	$(PYTHON) evaluate_correction.py --decoders="$(DATA)/models/phoc_decoder.$(DIM)d.$(MAX_WORDS).*.pt" --output=$(RESULTS)/correction.json --device=$(DEVICE)

report: baselines
	@$(PYTHON) report.py --states="$(DATA)/models/phoc_semantic.$(DIM)d.$(MAX_WORDS).*.pt" --baselines=$(RESULTS)

clean:
	rm -rf $(DATA)/models $(DATA)/results

mrproper:
	@case "$(abspath $(DATA))" in /|$(HOME)|$(abspath .)|$(abspath ..)) echo "refusing to remove DATA=$(DATA)" >&2; exit 1;; esac
	rm -rf $(DATA)
