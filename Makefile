.PHONY: lint test

lint:
	poetry run black --check benchmarking model-preparation
	poetry run flake8 benchmarking model-preparation
	poetry run mypy benchmarking model-preparation

test:
	poetry run pytest -n 32
