.PHONY: setup generate validate test evaluate

setup:
	uv sync --extra dev

generate:
	uv run python scripts/generate_reports.py

validate:
	uv run python scripts/validate_artifacts.py

test:
	uv run pytest

evaluate:
	uv run python scripts/evaluate.py
