black:
	black embetter tests setup.py --check

flake:
	flake8 embetter tests setup.py

test:
	pytest

install:
	python -m pip install -e ".[dev]"
	pre-commit install

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 tests
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 embetter

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: black flake interrogate test
