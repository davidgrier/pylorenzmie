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
                if code:
                    missing = MISSING.search(error)
                    if missing and missing.group(1) in OPTIONAL:
                        self.skipTest(f'needs {missing.group(1)}')
                self.assertEqual(code, 0, error)

    def test_the_modules_are_found(self):
        found = modules()
        for name in ('pylorenzmie.theory.Sphere',
                     'pylorenzmie.analysis.Frame',
                     'pylorenzmie.lmtool.LMTool'):
            self.assertIn(name, found)
        self.assertNotIn('pylorenzmie.lmtool.__main__', found)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
