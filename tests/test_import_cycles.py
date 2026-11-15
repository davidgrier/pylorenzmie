'''Every module can be the first one that a program imports.

The tests import ``pylorenzmie.lib`` first, which would hide a cycle
between the packages.  Each module is imported here in an interpreter of
its own, as a script would, and the interpreters run side by side.  A
module that needs an optional library that is not installed (CuPy, JAX,
Torch, Triton, Qt) is skipped.
'''
import importlib
import os
import pkgutil
import re
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

PACKAGES = ('pylorenzmie', 'pylorenzmie.lib', 'pylorenzmie.theory',
            'pylorenzmie.analysis', 'pylorenzmie.utilities',
            'pylorenzmie.lmtool')
OPTIONAL = ('cupy', 'jax', 'jaxlib', 'torch', 'triton', 'numba', 'tables',
            'PyQt5', 'PyQt6', 'PySide6')
MISSING = re.compile(r"No module named '(\w+)")
#: What pyqtgraph says when there is no Qt binding for it to use.
NO_QT = 'PyQtGraph requires one of'


def modules() -> list[str]:
    '''Return the names of the modules of the packages.

    ``__main__`` modules run a program when they are imported.
    '''
    found = []
    for name in PACKAGES:
        found.append(name)
        package = importlib.import_module(name)
        if name == 'pylorenzmie':
            continue
        walk = pkgutil.walk_packages(package.__path__, name + '.')
        found.extend(info.name for info in walk
                     if not info.name.endswith('.__main__'))
    return sorted(set(found))


def optional_missing(error: str) -> str | None:
    '''Return the optional library that an import failed for, if any.'''
    if NO_QT in error:
        return 'Qt'
    missing = MISSING.search(error)
    if missing and missing.group(1) in OPTIONAL:
        return missing.group(1)
    return None


def first_import(module: str) -> tuple[str, int, str]:
    '''Import a module in a fresh interpreter, and report how it went.'''
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    result = subprocess.run(
        [sys.executable, '-I', '-c', f'import {module}'],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return module, result.returncode, result.stderr


class TestFirstImport(unittest.TestCase):

    def test_each_module_imports_in_a_fresh_interpreter(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
            results = list(pool.map(first_import, modules()))
        for module, code, error in results:
            with self.subTest(module=module):
                if code and optional_missing(error):
                    self.skipTest(f'needs {optional_missing(error)}')
                self.assertEqual(code, 0, error)

    def test_a_missing_optional_library_is_told_from_a_failure(self):
        self.assertEqual(
            optional_missing("ModuleNotFoundError: No module named 'torch'"),
            'torch',
        )
        self.assertEqual(
            optional_missing('ImportError: PyQtGraph requires one of PyQt5, '
                             'PyQt6, PySide2 or PySide6'),
            'Qt',
        )
        self.assertIsNone(optional_missing(
            "ModuleNotFoundError: No module named 'pylorenzmie'"))
        self.assertIsNone(optional_missing(
            'ImportError: cannot import name x (circular import)'))

    def test_the_modules_are_found(self):
        found = modules()
        for name in ('pylorenzmie.theory.Sphere',
                     'pylorenzmie.analysis.Frame',
                     'pylorenzmie.lmtool.LMTool'):
            self.assertIn(name, found)
        self.assertNotIn('pylorenzmie.lmtool.__main__', found)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
