# Contributing to pylorenzmie

Thank you for helping. Bug reports, questions and pull requests are welcome
on [GitHub](https://github.com/davidgrier/pylorenzmie/issues). The
[architecture](https://pylorenzmie.readthedocs.io/en/latest/architecture.html),
[conventions](https://pylorenzmie.readthedocs.io/en/latest/conventions.html)
and [extending](https://pylorenzmie.readthedocs.io/en/latest/extending.html)
pages of the documentation explain how the package is put together.

## Setting up

Keep the virtual environment **outside** the repository (a venv inside the tree
is swept by bulk edits and deleted by `git clean -xdf`):

```sh
python3 -m venv ~/venvs/pylorenzmie
~/venvs/pylorenzmie/bin/pip install -e ".[dev,numba,docs]"   # editable install
~/venvs/pylorenzmie/bin/python -m pytest
```

Other commands: `python -m pytest --cov` (coverage), `python -m pytest -o
addopts=` (also the JAX tests), `ruff check .` (lint),
`pyright` (types) and `python -m sphinx -b html docs docs/_build/html` (the
documentation, in `docs/_build/html`).  `pre-commit install` (with `pip
install pre-commit`) runs ruff and the style tests before each commit.  The
optional `torch`, `jax` and `gpu` extras are not needed for the default tests.
To check the code without Numba, use a throwaway environment with only
`pip install -e ".[dev]"`.

## Style

* Python 3.11 or later. Lines are at most 79 characters.
* Single quotes, including in docstrings. American spelling.
* Docstrings are NumPy style, with `Parameters`, `Returns`, `Raises` and
  `Notes` as appropriate. Give the default value and the units of every
  parameter. Lengths are in micrometers, except coordinates in an image, which
  are in pixels (see the conventions page).
* Annotate the signatures. The package ships `py.typed`, so the annotations are
  part of the interface. `mypy` is advisory, not a requirement: it still
  reports problems that are internal to the package.

## Tests

Tests are in `tests/`, are run with pytest and are written with `unittest`.
Every change comes with tests: a fix with one that fails without it, a feature
with tests of its behavior and of its edge cases. Tests of code that needs
hardware or an optional package must skip when it is absent
(`unittest.skipUnless`), and the code that cannot be tested on the
continuous-integration machines must say so in its documentation. A numerical
change is checked against the NumPy calculator, which is the reference.

The suite takes about a minute.  A test that needs more than a few seconds
is a reason to make it smaller, not to move it: the tests that ran for a
minute each in a `tests/slow` directory did so for a search that their
assertions did not look at.  The JAX tests are the one set that the default
run leaves out, because JAX is optional; `python -m pytest -o addopts=` runs
them.  Continuous integration lints the code and the workflows, checks the
types, runs the tests on Python 3.11 to 3.14, with Numba and once without,
runs the whole suite with Qt on each binding and on macOS, with coverage in
one job (which fails below the threshold in `pyproject.toml`), runs the JAX
tests, installs the wheel outside the checkout, and builds the documentation
with warnings as errors.

## Documentation

Document every public class, method and function. The documentation build
checks every cross-reference (`nitpicky`), so a broken one fails it. Build it
with `python -m sphinx -b html docs docs/_build/html`, and run the examples in the docstrings; a doctest keeps an
example honest. A change that users can see goes into `CHANGELOG.md` under
*Unreleased*, and a breaking change says how to update.

## Commits and pull requests

* Commit messages follow
  [Conventional Commits](https://www.conventionalcommits.org/):
  `feat`, `fix`, `perf`, `docs`, `test`, `build`, `ci`, `refactor`, `chore`,
  with a `!` after the type for a breaking change. The first line is short; the
  body says why, and gives the numbers for a performance change.
* Keep a pull request to one subject, and say how you tested it. It is merged
  when continuous integration is green.
* Stage the files you changed by name. Do not `git add -A` a directory.

## Releasing (maintainers)

`pyproject.toml` holds the only copy of the version. To release: update
`CHANGELOG.md`, change `version` in `pyproject.toml`, push to `master`, wait for
continuous integration to pass, then tag and push the tag:

```sh
git tag -a vX.Y.Z -m "pylorenzmie X.Y.Z" && git push origin vX.Y.Z
```

The tag builds the package and publishes it to PyPI (`publish.yml`, which also
creates the GitHub release) and updates `CITATION.cff`
(`update-citation.yml`). A version cannot be published twice. The release is
archived on Zenodo, which mints a DOI for that version. `CITATION.cff` cites
the project (concept) DOI, which always resolves to the latest version, so it
needs no change at a release.

## License

pylorenzmie is distributed under the GNU General Public License, version 3 (see
`LICENSE`). By contributing, you agree to license your contribution under it.
Cite the software with `CITATION.cff`.
