.PHONY: setup check lint fmt test app

# Tools come from the local virtualenv. CI installs into the job's own
# environment and overrides this with an empty value: make check BIN=
BIN ?= .venv/bin/

setup:
	/opt/homebrew/Caskroom/miniconda/base/envs/main/bin/python3.11 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

check: lint test

lint:
	$(BIN)ruff check .
	$(BIN)ruff format --check .

fmt:
	$(BIN)ruff format .
	$(BIN)ruff check --fix .

test:
	$(BIN)pytest -q

app:
	PIPELINE_MODEL_HOST=$${PIPELINE_MODEL_HOST:-127.0.0.1} $(BIN)streamlit run src/pipeline_agents/app/streamlit_app.py
