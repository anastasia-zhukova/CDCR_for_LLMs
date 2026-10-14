# Run a CDCR experiment over every available dataset, one after another.
#
# Quick start:
#   make run-all                                  # sweep all datasets (ECB+ first)
#   make run-all ARGS="run=smoke"                 # pass Hydra overrides to every run
#   make run-all ARGS="approach=baseline_oracle"  # sweep the oracle baseline instead
#   make run-GVC                                  # run a single dataset
#   make list                                     # print the sweep order
#
# By default the sweep stops at the first failing dataset; re-running warm-starts
# the unfinished one from its checkpoint. Set KEEP_GOING=1 to push through failures
# and get a summary at the end instead.

# How to invoke the runner. Override e.g. `make PYTHON="python" run-all` if not using uv.
PYTHON ?= uv run python
SCRIPT := scripts/run_experiment.py
RUN    := $(PYTHON) $(SCRIPT)

# Extra Hydra overrides appended to every run (e.g. ARGS="llm.max_concurrency=10 run=smoke").
ARGS ?=

# Datasets known to the sweep, in run order: ECB+ first, then its META variant,
# then the rest. Each name is a config under configs/data/. Dual-kind datasets sweep
# mixed (_both, entity+event pooled and scored together); their single-kind configs
# (_entity, _event) live under configs/data/single/ and are run directly, e.g.
#   uv run python scripts/run_experiment.py data=single/ECBplus_entity
ALL_DATASETS := \
	ECBplus_both \
	ECBplusMETAm_both \
	CD2CR \
	CEREC \
	FCC \
	GVC \
	HyperCoref \
	MEANTIME_both \
	NewsWCL50_both \
	NiDENT \
	NP4E \
	WECEng

# Datasets to skip in the default sweep. They stay runnable individually
# (e.g. `make run-NiDENT`); add them back to the sweep with EXCLUDE="".
EXCLUDE ?= NiDENT NP4E

# What `run-all` actually sweeps: every known dataset minus the excluded ones.
DATASETS ?= $(filter-out $(EXCLUDE),$(ALL_DATASETS))

.DEFAULT_GOAL := help
.PHONY: help list run-all run-%

help:
	@echo "Targets:"
	@echo "  run-all            Run the experiment on every dataset (see 'make list'), sequentially."
	@echo "  run-<dataset>      Run a single dataset, e.g. make run-CD2CR."
	@echo "  list               Print the datasets in sweep order."
	@echo ""
	@echo "Variables:"
	@echo "  ARGS=\"...\"          Hydra overrides appended to every run."
	@echo "  KEEP_GOING=1        Continue past a failing dataset instead of stopping."
	@echo "  EXCLUDE=\"...\"       Datasets to skip in the default sweep (default: $(EXCLUDE))."
	@echo "  PYTHON=\"...\"        How to launch python (default: uv run python)."

list:
	@printf '%s\n' $(DATASETS)

# Sweep every dataset in order. Runs are sequential by construction (a single shell
# loop), so `make -j` can't reorder or parallelise them.
run-all:
	@failed=""; \
	for ds in $(DATASETS); do \
		echo "==> [$$ds] starting"; \
		$(RUN) data=$$ds $(ARGS); rc=$$?; \
		if [ $$rc -eq 0 ]; then \
			echo "==> [$$ds] done"; \
		else \
			echo "==> [$$ds] FAILED (exit $$rc)"; \
			failed="$$failed $$ds"; \
			if [ -z "$(KEEP_GOING)" ]; then \
				echo "Stopping. Fix the cause and re-run to warm-start from the checkpoint,"; \
				echo "or set KEEP_GOING=1 to continue past failures."; \
				exit $$rc; \
			fi; \
		fi; \
	done; \
	if [ -n "$$failed" ]; then echo "Failed datasets:$$failed"; exit 1; fi; \
	echo "All datasets completed."

# Run one dataset by config name, e.g. `make run-GVC`.
run-%:
	$(RUN) data=$* $(ARGS)
