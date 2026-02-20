.PHONY: lint format format-check test ci run bench

CONDA_ENV=llm

lint:
	conda run -n $(CONDA_ENV) ruff check .

format:
	conda run -n $(CONDA_ENV) ruff format .

format-check:
	conda run -n $(CONDA_ENV) ruff format --check .

test:
	conda run -n $(CONDA_ENV) pytest -q

ci: lint format-check test

run:
	conda run -n $(CONDA_ENV) ./scripts/server.sh

bench:
	conda run -n $(CONDA_ENV) ./scripts/bench.sh