.PHONY: install test lint format typecheck eval run demo-predictions gitleaks clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	.venv/bin/ruff check .
	.venv/bin/black --check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/black .

typecheck:
	.venv/bin/mypy --strict src/

eval:
	$(PY) -m eval.run

run:
	.venv/bin/uvicorn src.api.main:app --reload --port 8000

demo-predictions:
	$(PY) scripts/build_demo_predictions.py

gitleaks:
	gitleaks detect --no-banner --redact

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
