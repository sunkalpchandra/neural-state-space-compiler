.PHONY: install test lint format clean smoke

install:
	pip install -e ".[dev]"

test:
	pytest -q

test-fast:
	pytest -q -m "not slow"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts && ruff check --fix src tests scripts

smoke:
	nssc smoke

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
