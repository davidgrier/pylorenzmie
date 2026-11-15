'''qtpy must select the Qt binding before pyqtgraph can.

``qtpy`` honors ``QT_API``.  ``pyqtgraph`` chooses a binding of its own when
it is imported first, and ``qtpy`` then follows it, so ``QT_API`` is ignored.
Every module, script and example that imports ``pyqtgraph`` therefore
imports ``qtpy`` before it.  The check reads the text of every source file,
the scripts and examples that sit in strings and documents included.  It
needs no Qt binding.

Without a binding, the ``lmtool`` command says which extra to install.
'''
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = ('.py', '.rst', '.md')
SKIP = ('devel', 'build', '_build', 'htmlcov')
PYQTGRAPH = re.compile(r'^\s*(import pyqtgraph|from pyqtgraph)')
QTPY = re.compile(r'^\s*(import qtpy|from qtpy)')


def sources():
    '''Yield the source files of the repository, not scratch directories.'''
    for path in sorted(ROOT.rglob('*')):
        parts = path.relative_to(ROOT).parts
        if (path.suffix in SUFFIXES and path.is_file()
                and not any(part.startswith('.') or part in SKIP
                            or part.endswith('.egg-info')
                            for part in parts)):
            yield path


def first(pattern: re.Pattern, lines: list[str]) -> int | None:
    '''Return the number of the first line that matches *pattern*.'''
    for number, line in enumerate(lines, 1):
        if pattern.match(line):
            return number
    return None


class TestQtpyComesBeforePyqtgraph(unittest.TestCase):

    def test_no_file_imports_pyqtgraph_first(self):
        late = []
        for path in sources():
            lines = path.read_text(errors='replace').splitlines()
            graph = first(PYQTGRAPH, lines)
            if graph is None:
                continue
            select = first(QTPY, lines)
            if select is None or graph < select:
                late.append(f'{path.relative_to(ROOT)}:{graph}')
        self.assertEqual(late, [], 'import qtpy before pyqtgraph')

    def test_the_check_sees_a_late_import(self):
        lines = ['import pyqtgraph as pg', 'from qtpy import QtCore']
        self.assertLess(first(PYQTGRAPH, lines), first(QTPY, lines))
        self.assertIsNone(first(QTPY, ['import numpy']))

    def test_the_check_reads_the_files_of_lmtool(self):
        names = {path.name for path in sources()}
        self.assertIn('LMTool.py', names)
        self.assertIn('test_lmtool.py', names)


class TestWithoutABinding(unittest.TestCase):

    def test_the_command_names_the_extras_to_install(self):
        # a module that is None in sys.modules cannot be imported
        code = ('import sys\n'
                'sys.modules["qtpy"] = None\n'
                'import pylorenzmie.lmtool.launch\n')
        result = subprocess.run(
            [sys.executable, '-I', '-c', code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pylorenzmie[pyqt6]', result.stderr)
        self.assertIn('pyside6', result.stderr)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
