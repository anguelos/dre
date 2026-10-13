# Query augmentation: dre and baselines under Penha et al. query variations and document noise. REPORT.md is
# regenerated after every finished (method, dataset, version), so it fills in as results come.
#   make stage                WordNet and GloVe 6B into data/ (GLOVE_ZIP=path/to/glove.6B.zip avoids the download)
#   make augment              the query variants (data/re2rnn/*/rules.*.tsv, data/trec5/queries.*.tsv)
#   make re2rnn               all RE2RNN runs (CPU; make -j6 re2rnn)
#   make trec5_gpu            TREC-5 dre runs, one at a time, original documents first
#   make trec5_cpu            TREC-5 baselines (bm25, re; fuzzy on the original documents only)
#   make report
# A finished (method, dataset, version) leaves data/done/<benchmark>/<method>/<dataset>.<version>; run.py skips
# variants that already have a result, so an interrupted run resumes; the inputs are order-only prerequisites (a
# regenerated query file does not redo finished runs: delete data/done and data/results to start over).
PYTHON ?= python3
PYTHONPATH := ../../src:$(PYTHONPATH)
export PYTHONPATH
DEVICE ?= cpu
GPU_DEVICE ?= cuda
WORKERS ?= 6
DATA := data
DIM ?= 100
ENGINE ?= viterbi

RE2RNN_DATASETS ?= ATIS TREC SMS
RE2RNN_VERSIONS ?= clean noise5 noise20
RE2RNN_METHODS ?= exact.$(ENGINE) levenshtein.$(ENGINE) phoc.$(ENGINE) cosine$(DIM).$(ENGINE) csls$(DIM).$(ENGINE) phoc_csls$(DIM).$(ENGINE)
TREC5_VERSIONS ?= original degrade5 degrade20
TREC5_DRE ?= exact.$(ENGINE) phoc.$(ENGINE) levenshtein.$(ENGINE) phoc_csls$(DIM).$(ENGINE)
TREC5_BASELINES ?= bm25_word bm25_char4 re

AUGMENTED := $(DATA)/trec5/queries.original.tsv
STAGED := $(DATA)/nltk_data/corpora/wordnet.zip $(DATA)/glove/glove$(DIM).pt
GLOVE_ZIP ?=

.PHONY: all stage augment re2rnn trec5_gpu trec5_cpu report
all: re2rnn trec5_gpu trec5_cpu

stage: $(STAGED)
$(STAGED): stage.py
	$(PYTHON) stage.py --dims=$(DIM) $(if $(GLOVE_ZIP),--glove_zip=$(GLOVE_ZIP))

augment: $(AUGMENTED)
$(AUGMENTED): augment.py $(DATA)/nltk_data/corpora/wordnet.zip
	$(PYTHON) augment.py

report:
	$(PYTHON) report.py

# $(1) benchmark, $(2) method, $(3) dataset, $(4) version, $(5) device
define RUN
$(DATA)/done/$(1)/$(2)/$(3).$(4): | $(AUGMENTED) $(DATA)/glove/glove$(DIM).pt
	$(PYTHON) run.py --benchmark=$(1) --method=$(2) --dataset=$(3) --version=$(4) --device=$(5) --workers=$(WORKERS)
	@mkdir -p $$(@D) && touch $$@
	-$(PYTHON) report.py
endef

$(foreach m,$(RE2RNN_METHODS),$(foreach d,$(RE2RNN_DATASETS),$(foreach v,$(RE2RNN_VERSIONS),$(eval $(call RUN,re2rnn,$(m),$(d),$(v),$(DEVICE))))))
re2rnn: $(foreach v,$(RE2RNN_VERSIONS),$(foreach m,$(RE2RNN_METHODS),$(foreach d,$(RE2RNN_DATASETS),$(DATA)/done/re2rnn/$(m)/$(d).$(v))))

$(foreach m,$(TREC5_DRE),$(foreach v,$(TREC5_VERSIONS),$(eval $(call RUN,trec5,$(m),trec5,$(v),$(GPU_DEVICE)))))
trec5_gpu: $(foreach v,$(TREC5_VERSIONS),$(foreach m,$(TREC5_DRE),$(DATA)/done/trec5/$(m)/trec5.$(v)))

$(foreach m,$(TREC5_BASELINES),$(foreach v,$(TREC5_VERSIONS),$(eval $(call RUN,trec5,$(m),trec5,$(v),cpu))))
$(eval $(call RUN,trec5,fuzzy,trec5,original,cpu))
trec5_cpu: $(foreach v,$(TREC5_VERSIONS),$(foreach m,$(TREC5_BASELINES),$(DATA)/done/trec5/$(m)/trec5.$(v))) $(DATA)/done/trec5/fuzzy/trec5.original
