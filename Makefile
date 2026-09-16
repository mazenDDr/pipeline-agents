.PHONY: setup check lint fmt test

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
