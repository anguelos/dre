.PHONY: clean test coverage doc build install

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .coverage coverage.xml docs/_build tmp
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest test/unit_coverage test/bugfixing test/corner_cases

coverage:
	pytest test/unit_coverage --cov=src/dre --cov-report=term-missing

doc:
	sphinx-build -b html docs docs/_build/html

build:
	python -m build

install:
	pip install -e .
