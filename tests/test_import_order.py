'''The imports of a module come in groups, in a fixed order.

The standard library comes first, then the third-party libraries
(``dgimage`` and ``pyqtgraph`` among them), then ``pylorenzmie``, and last
the siblings of a package (``from .name import``) or of a test.  A blank
line separates the groups.  Absolute imports are the rule between the
packages, so that a parent is never imported with dots (ruff ``TID252``).
The check reads every Python file of the repository, the tests and the
tools included.
'''
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = ('devel', 'build', '_build', 'htmlcov')

FUTURE, STDLIB, THIRD, OWN, LOCAL = range(5)
GROUPS = 'future standard-library third-party pylorenzmie local'.split()
#: The modules that sit beside the tests.
HELPERS = {path.stem for path in (ROOT / 'tests').glob('*.py')}


def sources():
    '''Yield the Python files of the repository, not scratch directories.'''
    for path in sorted(ROOT.rglob('*.py')):
        parts = path.relative_to(ROOT).parts
        if not any(part.startswith('.') or part in SKIP
                   or part.endswith('.egg-info') for part in parts):
            yield path


def group(node: ast.Import | ast.ImportFrom) -> int:
    '''Return the group of an import: where it belongs in the block.'''
    if isinstance(node, ast.ImportFrom) and node.level:
        return LOCAL
    name = (node.module if isinstance(node, ast.ImportFrom)
            else node.names[0].name) or ''
    top = name.split('.')[0]
    if top in HELPERS:
        return LOCAL
    if top == '__future__':
        return FUTURE
    if top in sys.stdlib_module_names:
        return STDLIB
    return OWN if top == 'pylorenzmie' else THIRD


def misplaced(source: str) -> tuple[int, str, str] | None:
    '''Return the first import of the top block that is in the wrong group.

    The block is the imports at the head of the module, before the first
    statement that does something (a docstring, a ``try`` or an ``if`` that
    guards an import do not end it).  Returns ``(line, group, after)``.
    '''
    last = FUTURE
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            here = group(node)
            if here < last:
                return node.lineno, GROUPS[here], GROUPS[last]
            last = here
        elif not (isinstance(node, ast.Try | ast.If) or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant))):
            return None
    return None


class TestTheImportsAreInGroups(unittest.TestCase):

    def test_no_module_has_a_group_out_of_place(self):
        late = []
        for path in sources():
            found = misplaced(path.read_text())
            if found:
                line, here, after = found
                late.append(f'{path.relative_to(ROOT)}:{line} '
                            f'{here} after {after}')
        self.assertEqual(late, [], 'imports out of order')

    def test_the_check_sees_a_group_out_of_place(self):
        self.assertEqual(
            misplaced('import numpy\nimport os\n'),
            (2, 'standard-library', 'third-party'),
        )
        self.assertEqual(
            misplaced('import os\nimport numpy\nimport pylorenzmie\n'
                      'from . import z\n'),
            None,
        )
        self.assertEqual(
            misplaced('from pylorenzmie import x\nimport numpy\n'),
            (2, 'third-party', 'pylorenzmie'),
        )

    def test_a_statement_ends_the_block(self):
        self.assertIsNone(misplaced('import numpy\nx = 1\nimport os\n'))

    def test_a_docstring_and_a_guarded_import_do_not(self):
        text = ("'''Doc.'''\nimport os\ntry:\n    import numpy\n"
                "except ImportError:\n    pass\nimport pylorenzmie\n")
        self.assertIsNone(misplaced(text))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
