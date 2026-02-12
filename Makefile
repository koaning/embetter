.PHONY: docs

ruff:
	uvx ruff check embetter tests --fix

test:
	uv run pytest -n auto -vv

install:
	uv venv
	uv pip install -e ".[dev]"

pypi:
	uv build
	uv publish

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: clean ruff test clean

docs:
	cp README.md docs/index.md
	uv run mkdocs serve

deploy-docs:
	cp README.md docs/index.md
	uv run mkdocs gh-deploy
