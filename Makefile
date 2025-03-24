.PHONY: docs

ruff: 
	python -m ruff check embetter tests setup.py --fix

test:
	pytest -n auto -vv

install:
	python -m pip install -e ".[dev]"

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: clean ruff test clean

docs:
	cp README.md docs/index.md
	python -m mkdocs serve

deploy-docs:
	cp README.md docs/index.md
	python -m mkdocs gh-deploy
