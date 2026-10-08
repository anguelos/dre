# Zero-shot rule-based classification mirroring RE2RNN / FA-RNN (Jiang et al., EMNLP 2020) with dre.
# Run from examples/semantics; `make help` lists the targets. Methods are discovered from run_<method>.py scripts.
PYTHON ?= python3
DATA ?= data
DATASETS ?= ATIS TREC SMS
VERSIONS ?= clean noise5 noise20
DEVICE ?= cpu
export PYTHONPATH := ../../src$(if $(PYTHONPATH),:$(PYTHONPATH))

METHODS := $(sort $(patsubst run_%.py,%,$(filter-out run_common.py,$(wildcard run_*.py))))
SPLITS := $(foreach d,$(DATASETS),$(foreach v,$(VERSIONS),$(d).$(v)))
COMMON := run_common.py re2rnn_rules.py $(wildcard ../../src/dre/*.py) $(wildcard ../../src/phoc/*.py)
dataset = $(word 1,$(subst ., ,$(1)))
version = $(word 2,$(subst ., ,$(1)))

.SECONDARY:
.DELETE_ON_ERROR:
.PHONY: help download compile runs evals report clean mrproper $(foreach m,$(METHODS),run_$(m) eval_$(m) clean_$(m))

help:
	@echo "make download        fetch RE2RNN's data archive (~18 MB) into $(DATA)/raw; never triggered implicitly"
	@echo "make compile         datasets, noisy test versions, converted rules and GloVe matrices"
	@echo "make run_<method>    predictions for DATASETS=\"$(DATASETS)\" x VERSIONS=\"$(VERSIONS)\""
	@echo "make eval_<method>   accuracy of those predictions; make clean_<method> removes them"
	@echo "make runs, evals     every method; make report prints the accuracy table"
	@echo "make clean           remove predictions and evaluations (downloads and compiled data are kept)"
	@echo "make mrproper        remove everything under $(DATA)"
	@echo "methods: $(METHODS)"

download:
	$(PYTHON) stage_re2rnn.py --data_root=$(DATA)

$(DATA)/raw/SHA256SUMS:
	@echo "$@ is missing: run 'make download' first" >&2; exit 1

$(DATA)/.compiled: $(DATA)/raw/SHA256SUMS compile_re2rnn.py re2rnn_rules.py
	$(PYTHON) compile_re2rnn.py --data_root=$(DATA)

compile: $(DATA)/.compiled

define METHOD_RULES
run_$(1): $(foreach s,$(SPLITS),$(DATA)/runs/$(1)/$(s).tsv)
eval_$(1): $(foreach s,$(SPLITS),$(DATA)/eval/$(1)/$(s).json)
clean_$(1):
	rm -rf $(DATA)/runs/$(1) $(DATA)/eval/$(1)
$(DATA)/runs/$(1)/%.tsv: $(DATA)/.compiled run_$(1).py $(COMMON)
	$(PYTHON) run_$(1).py --data_root=$(DATA) --dataset=$$(call dataset,$$*) --version=$$(call version,$$*) --output=$$@ --device=$(DEVICE)
$(DATA)/eval/$(1)/%.json: $(DATA)/runs/$(1)/%.tsv evaluate.py
	@mkdir -p $$(dir $$@)
	$(PYTHON) evaluate.py --predictions=$$< --output=$$@
endef
$(foreach m,$(METHODS),$(eval $(call METHOD_RULES,$(m))))

runs: $(addprefix run_,$(METHODS))
evals: $(addprefix eval_,$(METHODS))

report: evals
	@$(PYTHON) report.py --data_root=$(DATA) --datasets="$(DATASETS)" --versions="$(VERSIONS)"

clean:
	rm -rf $(DATA)/runs $(DATA)/eval

mrproper:
	@case "$(abspath $(DATA))" in /|$(HOME)|$(abspath .)|$(abspath ..)) echo "refusing to remove DATA=$(DATA)" >&2; exit 1;; esac
	rm -rf $(DATA)
