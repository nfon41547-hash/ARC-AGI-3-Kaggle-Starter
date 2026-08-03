PYTHON          ?= python3.12
VENV            := .venv
VENV_PY         := $(VENV)/bin/python
VENV_PIP        := $(VENV)/bin/pip
KAGGLE          := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle
FRAMEWORK_REPO  := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR   := vendor/ARC-AGI-3-Agents
GAME            ?=
STEPS           ?= 200

.PHONY: help setup verify-sovereign play-local notebook submit status clean _check-kaggle

_check-kaggle:
	@if [ ! -s .kaggle/access_token ]; then echo "ERROR: missing .kaggle/access_token"; exit 1; fi

help:
	@echo "make setup | verify-sovereign | play-local | notebook | submit | status | clean"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "arc-agi==0.9.9" "arcengine==0.9.3" "numpy>=1.26,<3" "kaggle>=2.2" python-dotenv pandas pyarrow pytest
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); else git -C $(FRAMEWORK_DIR) pull --ff-only; fi
	@$(VENV_PY) scripts/slim_framework.py
	@$(VENV_PY) scripts/verify_sovereign.py

verify-sovereign:
	$(VENV_PY) scripts/verify_sovereign.py
	$(VENV_PY) -m pytest -q tests/test_sovereign_runtime.py

play-local: verify-sovereign
	$(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

notebook: verify-sovereign
	$(VENV_PY) scripts/build_notebook.py

submit: notebook _check-kaggle
	$(KAGGLE) kernels push -p notebooks/

status: _check-kaggle
	@KERNEL_ID=$$($(PYTHON) -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); $(KAGGLE) kernels status $$KERNEL_ID

clean:
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb reference logs.log __pycache__ .pytest_cache
