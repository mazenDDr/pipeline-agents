.PHONY: setup check lint fmt test app

setup:
	/opt/homebrew/Caskroom/miniconda/base/envs/main/bin/python3.11 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

check: lint test

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

test:
	.venv/bin/pytest -q

app:
	PIPELINE_MODEL_HOST=$${PIPELINE_MODEL_HOST:-127.0.0.1} .venv/bin/streamlit run src/pipeline_agents/app/streamlit_app.py
