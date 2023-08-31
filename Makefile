.PHONY: docs

black:
	black embetter tests setup.py

flake:
	flake8 embetter tests setup.py

test:
	pytest -n auto -vv

install:
	python -m pip install -e ".[dev]"

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 tests
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 embetter

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: clean black flake interrogate test clean

docs:
	cp README.md docs/index.md
	python -m mkdocs serve

deploy-docs:
	cp README.md docs/index.md
	python -m mkdocs gh-deploy
