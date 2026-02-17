.PHONY: lint format test ci

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

ci: lint test
