# ARC Prize 2026 — ARC-AGI-3 Pure NumPy development workflow.
#
# Core loop:
#   make setup                # Python 3.12 environment + official framework
#   make test                 # deterministic regression and intelligence gates
#   make benchmark            # 600-episode adaptive-vs-random benchmark
#   make play-local           # run against locally available official games
#   make notebook             # build the offline Kaggle notebook
#   make submit               # push the notebook to Kaggle

PYTHON          ?= python3.12
VENV            := .venv
VENV_PY         := $(VENV)/bin/python
VENV_PIP        := $(VENV)/bin/pip
KAGGLE          := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle
FRAMEWORK_REPO  := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR   := vendor/ARC-AGI-3-Agents
COMP_SLUG       := arc-prize-2026-arc-agi-3
GAME            ?=
STEPS           ?= 200

.PHONY: help setup test benchmark play-local pull-sample notebook submit status \
        verify-local verify-intelligence clean _check-kaggle

_check-kaggle:
	@if [ ! -s .kaggle/access_token ]; then \
	    echo "ERROR: .kaggle/access_token is missing or empty."; \
	    echo "       Generate a fresh token at https://www.kaggle.com/settings"; \
	    echo "       and save it as a one-line file at: $(PWD)/.kaggle/access_token"; \
	    exit 1; \
	fi

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Vars: PYTHON=$(PYTHON)  GAME=$(GAME)  STEPS=$(STEPS)"

setup: ## One-time install: pinned toolkit, NumPy, tests, Kaggle CLI, framework
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install \
	    "arc-agi==0.9.9" \
	    "arcengine==0.9.3" \
	    "numpy>=2.0,<3" \
	    "pytest>=8,<9" \
	    "kaggle>=2.2" \
	    python-dotenv pandas pyarrow
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then \
	    mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); \
	else \
	    git -C $(FRAMEWORK_DIR) pull --ff-only; \
	fi
	@$(VENV_PY) scripts/slim_framework.py
	@echo ""
	@echo "Setup complete. Run: make test && make benchmark"

test: ## Run legality, determinism, adaptation, and safety regression gates
	$(VENV_PY) -m py_compile agent/my_agent.py scripts/benchmark_agent.py
	$(VENV_PY) -m pytest -q

benchmark: ## Rebuild deterministic v42 intelligence evidence
	$(VENV_PY) scripts/benchmark_agent.py

verify-intelligence: test benchmark ## Full local evidence gate before official play
	@$(VENV_PY) -c "import json; p=json.load(open('evidence/intelligence_benchmark_v42.json')); assert all(c['adaptive']['illegal_actions']==0 for c in p['conditions']); print('intelligence gate: PASS')"

play-local: ## Run against all official local games, or GAME=ls20
	$(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify-local: verify-intelligence ## Intelligence gates plus 50-step official smoke test
	$(VENV_PY) scripts/play_local.py --game ls20,vc33 --max-steps 50

list-games: ## Show all available official local games
	$(VENV_PY) scripts/play_local.py --list

pull-sample: _check-kaggle ## Download the official sample notebook for reference
	mkdir -p reference/stochastic-goose
	$(KAGGLE) kernels pull inversion/arc3-sample-submission-stochastic-goose \
	    -p reference/stochastic-goose -m
	@echo "Open reference/stochastic-goose/*.ipynb for the canonical pattern."

notebook: verify-intelligence ## Build notebooks/submission.ipynb after gates pass
	$(VENV_PY) scripts/build_notebook.py

submit: notebook _check-kaggle ## Build, verify, and push the notebook to Kaggle
	@grep -q REPLACE_WITH_YOUR_USERNAME notebooks/kernel-metadata.json && { \
	    echo "ERROR: edit notebooks/kernel-metadata.json and replace REPLACE_WITH_YOUR_USERNAME"; \
	    exit 1; } || true
	$(KAGGLE) kernels push -p notebooks/
	@echo ""
	@echo "Pushed. Track it with: make status"

status: _check-kaggle ## Show the status of the most recent Kaggle kernel run
	@KERNEL_ID=$$($(PYTHON) -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

clean: ## Remove generated local artifacts; never remove source/evidence
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb \
	       reference logs.log __pycache__ .pytest_cache tests/__pycache__ \
	       agent/__pycache__ scripts/__pycache__
