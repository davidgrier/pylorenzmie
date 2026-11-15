'''Every package can be the first one that a program imports.

The tests import ``pylorenzmie.lib`` first, which would hide a cycle
between the packages.  Each module is imported here in an interpreter of
its own, as a script would.
'''
import subprocess
import sys
import unittest

MODULES = ('pylorenzmie', 'pylorenzmie.lib', 'pylorenzmie.theory',
           'pylorenzmie.analysis', 'pylorenzmie.utilities',
           'pylorenzmie.theory.Sphere', 'pylorenzmie.theory.Aberrated',
           'pylorenzmie.analysis.Feature', 'pylorenzmie.analysis.Frame',
           'pylorenzmie.analysis.Optimizer')


class TestFirstImport(unittest.TestCase):

    def test_each_module_imports_in_a_fresh_interpreter(self):
        for module in MODULES:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, '-I', '-c', f'import {module}'],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
