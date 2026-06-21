#!/usr/bin/env python3
'''Update the pylorenzmie @software entry in ~/texmf/bibtex/bib/grier.bib.

Reads version, date, DOI, and abstract from CITATION.cff at the repo root
and rewrites the @software{pylorenzmie, ...} block in grier.bib.  If no
such block exists yet, it is prepended to the file.

Run directly after a release pull, or install as a post-merge git hook:

    ln -sf ../../scripts/update_bib.py .git/hooks/post-merge
    chmod +x .git/hooks/post-merge
'''

import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CITATION = REPO / 'CITATION.cff'
BIB = Path.home() / 'texmf/bibtex/bib/grier.bib'

MONTHS = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
]


def _parse_citation(path: Path) -> dict:
    '''Extract scalar and block-scalar fields from CITATION.cff without PyYAML.'''
    text = path.read_text()
    fields = {}

    for key in ('version', 'date-released', 'doi'):
        m = re.search(rf'^{key}:\s*["\']?([^"\'\n]+)["\']?', text, re.MULTILINE)
        if m:
            fields[key] = m.group(1).strip()

    # abstract uses a YAML block scalar (>-)
    m = re.search(r'^abstract:\s*>-?\n((?:[ \t]+.+\n?)+)', text, re.MULTILINE)
    if m:
        lines = [l.strip() for l in m.group(1).splitlines()]
        fields['abstract'] = ' '.join(lines)

    return fields


def _make_entry(fields: dict) -> str:
    version = fields.get('version', '')
    doi = fields.get('doi', '')
    abstract = fields.get('abstract', '')
    date_str = fields.get('date-released', '')
    dt = datetime.fromisoformat(date_str) if date_str else datetime.now()
    month = MONTHS[dt.month - 1]
    year = dt.year

    return (
        '@software{pylorenzmie,\n'
        f'  author =\t\t {{Grier, David G.}},\n'
        f'  title =\t\t {{pylorenzmie}},\n'
        f'  month =\t\t {month},\n'
        f'  year =\t\t {year},\n'
        f'  publisher =\t {{Zenodo}},\n'
        f'  version =\t\t {{{version}}},\n'
        f'  doi =\t\t\t {{{doi}}},\n'
        f'  url =\t\t\t {{https://doi.org/{doi}}},\n'
        f'  abstract =\t {{{abstract}}},\n'
        f'  github =\t\t {{https://github.com/davidgrier/pylorenzmie}},\n'
        f'  documentation =\t {{https://pylorenzmie.readthedocs.io/}},\n'
        f'  pypi =\t\t {{https://pypi.org/project/pylorenzmie/}},\n'
        '}'
    )


def main() -> None:
    if not CITATION.exists():
        print(f'update_bib: CITATION.cff not found at {CITATION}', file=sys.stderr)
        sys.exit(1)
    if not BIB.exists():
        print(f'update_bib: grier.bib not found at {BIB}', file=sys.stderr)
        sys.exit(1)

    fields = _parse_citation(CITATION)
    if not fields.get('version'):
        sys.exit(0)  # CITATION.cff not yet populated; nothing to do

    entry = _make_entry(fields)
    text = BIB.read_text()

    pattern = re.compile(r'@software\{pylorenzmie,.*?\n\}', re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(entry, text)
    else:
        text = entry + '\n\n' + text

    BIB.write_text(text)
    print(f"update_bib: pylorenzmie {fields.get('version')} → {BIB}")


if __name__ == '__main__':
    main()
