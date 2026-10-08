# IR experiments for dre. Run from examples/ir; `make help` lists the targets.
# Methods are discovered from run_<method>.py scripts: each gets run_<method>, eval_<method> and clean_<method>.
PYTHON ?= python3
DATA ?= data
COLLECTION ?= trec5
VERSIONS ?= original degrade5 degrade20
REFERENCE ?= original
# Scripts parallelise internally over documents; with `make -j`, lower WORKERS to avoid oversubscribing the cores.
WORKERS ?= $(shell nproc)
export PYTHONPATH := ../../src$(if $(PYTHONPATH),:$(PYTHONPATH))
# PyTerrier's Java bridge needs JAVA_HOME; derive it from the java on the PATH (a JRE is enough, no JDK needed).
export JAVA_HOME ?= $(patsubst %/bin/java,%,$(realpath $(shell which java)))

C := $(DATA)/$(COLLECTION)
ALL_VERSIONS := original degrade5 degrade20
METHODS := $(sort $(patsubst run_%.py,%,$(wildcard run_*.py)))
# The baselines the paper compares against; only those whose run_<method>.py exists are built.
BASELINES := $(filter re fuzzy bm25_word bm25_char4 bm25_char5,$(METHODS))
COMMON := trec_io.py word_regex.py

# Never delete intermediate files of the chain, and never leave a half-written target behind.
.SECONDARY:
.DELETE_ON_ERROR:
.PHONY: help download compile runs evals baselines eval_baselines clean_baselines clean mrproper $(foreach m,$(METHODS),run_$(m) eval_$(m) clean_$(m))

help:
	@echo "make download        stage $(COLLECTION) into $(C)/raw (may use the internet; never triggered implicitly)"
	@echo "make compile         documents, tokens, queries and qrels from the downloads"
	@echo "make run_<method>    runs of one method for VERSIONS=\"$(VERSIONS)\""
	@echo "make eval_<method>   evaluate one method's runs against the qrels"
	@echo "make clean_<method>  remove one method's runs and evaluations"
	@echo "make runs, evals     the same for every method"
	@echo "make baselines       runs of every baseline; also eval_baselines, clean_baselines"
	@echo "make clean           remove all runs and evaluations; downloads and compiled data are kept"
	@echo "make mrproper        remove everything under $(DATA), downloads included"
	@echo "methods: $(METHODS)"
	@echo "baselines: $(BASELINES)"
	@echo "variables: COLLECTION=$(COLLECTION) DATA=$(DATA) VERSIONS=\"$(VERSIONS)\" REFERENCE=$(REFERENCE) PYTHON=$(PYTHON) WORKERS=$(WORKERS)"

download:
	$(PYTHON) stage_$(COLLECTION).py --data_root=$(DATA) -v

$(C)/raw/SHA256SUMS:
	@echo "$@ is missing: run 'make download COLLECTION=$(COLLECTION)' first" >&2; exit 1

# compile_<collection>.py writes every version at once; the stamp gives that multi-file step a single target.
$(C)/docs/.stamp: $(C)/raw/SHA256SUMS compile_$(COLLECTION).py $(COMMON)
	$(PYTHON) compile_$(COLLECTION).py --data_root=$(DATA) --workers=$(WORKERS) -v
	touch $@

$(C)/tokens/%.tsv: $(C)/docs/.stamp compile_tokens.py $(COMMON)
	$(PYTHON) compile_tokens.py --data_root=$(DATA) --collection=$(COLLECTION) --version=$* --workers=$(WORKERS)

$(C)/queries.tsv: $(C)/tokens/$(REFERENCE).tsv compile_queries.py $(COMMON)
	$(PYTHON) compile_queries.py --data_root=$(DATA) --collection=$(COLLECTION) --reference=$(REFERENCE) -v

$(C)/qrels.txt: $(C)/queries.tsv $(C)/tokens/$(REFERENCE).tsv compile_qrels.py $(COMMON)
	$(PYTHON) compile_qrels.py --data_root=$(DATA) --collection=$(COLLECTION) --reference=$(REFERENCE) --workers=$(WORKERS)

compile: $(foreach v,$(ALL_VERSIONS),$(C)/tokens/$(v).tsv) $(C)/queries.tsv $(C)/qrels.txt

# A method may declare extra prerequisites (e.g. an index) in DEPS_<method>; the dre methods depend on the library.
DRE_SOURCES := dre_run.py $(wildcard ../../src/dre/*.py)
DEPS_dre_exact := $(DRE_SOURCES)
DEPS_phoc_soft := $(DRE_SOURCES) $(wildcard ../../src/phoc/*.py)
DEPS_levenshtein_soft := $(DRE_SOURCES)
DEPS_bm25_word := bm25_run.py
DEPS_bm25_char4 := bm25_run.py
DEPS_bm25_char5 := bm25_run.py
define METHOD_RULES
run_$(1): $(foreach v,$(VERSIONS),$(C)/runs/$(1)/$(v).run)
eval_$(1): $(foreach v,$(VERSIONS),$(C)/eval/$(1)/$(v).tsv)
clean_$(1):
	rm -rf $(C)/runs/$(1) $(C)/eval/$(1)
$(C)/runs/$(1)/%.run: $(C)/tokens/%.tsv $(C)/queries.tsv run_$(1).py $(COMMON) $(DEPS_$(1))
	$(PYTHON) run_$(1).py --data_root=$(DATA) --collection=$(COLLECTION) --version=$$* --output=$$@ --workers=$(WORKERS)
$(C)/eval/$(1)/%.tsv: $(C)/runs/$(1)/%.run $(C)/qrels.txt evaluate.py trec_io.py
	$(PYTHON) evaluate.py --qrels=$(C)/qrels.txt --queries=$(C)/queries.tsv --run=$$< --output=$$@ -v
endef
$(foreach m,$(METHODS),$(eval $(call METHOD_RULES,$(m))))

runs: $(addprefix run_,$(METHODS))
evals: $(addprefix eval_,$(METHODS))
baselines: $(addprefix run_,$(BASELINES))
eval_baselines: $(addprefix eval_,$(BASELINES))
clean_baselines: $(addprefix clean_,$(BASELINES))

clean:
	rm -rf $(DATA)/*/runs $(DATA)/*/eval

mrproper:
	@case "$(abspath $(DATA))" in /|$(HOME)|$(abspath .)|$(abspath ..)) echo "refusing to remove DATA=$(DATA)" >&2; exit 1;; esac
	rm -rf $(DATA)
