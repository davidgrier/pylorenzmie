# Developer shortcuts. The commands are the ones that CI runs
# (see .github/workflows/test.yml) and that CLAUDE.md documents.

PYTHON ?= python

.PHONY: all help dev test test-all cov docs clean

all: test

help:
	@echo 'make dev       install the package in editable mode with the test, Numba and docs extras'
	@echo 'make test      run the test suite, as CI does (without tests/slow and the JAX tests)'
	@echo 'make test-all  run all tests, including tests/slow and the JAX tests'
	@echo 'make cov       run the test suite and report coverage'
	@echo 'make docs      build the HTML documentation in docs/_build/html'
	@echo 'make clean     remove build, cache and coverage files'
	@echo
	@echo 'Variable: PYTHON (default: python)'

# Run this with the virtual environment's Python, e.g.
# make dev PYTHON=~/python/venvs/pylorenzmie/bin/python
dev:
	$(PYTHON) -m pip install -e ".[dev,numba,docs]"

test:
	$(PYTHON) -m pytest --tb=short -q

# pyproject.toml excludes tests/slow and the JAX tests by default
test-all:
	$(PYTHON) -m pytest -o addopts= --tb=short -q

# Numba-compiled functions run as machine code, which coverage cannot see.
# NUMBA_DISABLE_JIT=1 make cov  measures their Python source (slowly).
cov:
	$(PYTHON) -m pytest --cov --tb=short -q

docs:
	$(PYTHON) -m sphinx -b html docs docs/_build/html

clean:
	rm -rf .pytest_cache .coverage htmlcov docs/_build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . \( -name '*.nbi' -o -name '*.nbc' \) -delete
