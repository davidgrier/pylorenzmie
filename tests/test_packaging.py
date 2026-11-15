'''The package list in pyproject.toml is the list of the packages.

setuptools puts into the wheel only the packages that are listed, so a
package that is missing from the list works in a checkout and is absent
from what ``pip install`` installs.
'''
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOT_PACKAGES = ('tests', 'docs', 'tools', 'devel', 'build', 'htmlcov')

#: Directories that are packages without an ``__init__.py``: ``lmtool``
#: holds the program, which is launched by name and never imported as a
#: library.
NAMESPACES = ('lmtool',)


def configuration() -> dict:
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)['tool']['setuptools']


def package_directories() -> set[str]:
    '''Return the dotted names of the directories that are packages.'''
    found = set()
    for path in ROOT.rglob('__init__.py'):
        relative = path.parent.relative_to(ROOT).parts
        if relative and (relative[0] in NOT_PACKAGES
                         or relative[0].startswith('.')):
            continue
        found.add('.'.join(('pylorenzmie',) + relative))
    found.update(f'pylorenzmie.{name}' for name in NAMESPACES)
    return found


class TestPackaging(unittest.TestCase):

    def test_every_package_is_listed(self):
        listed = set(configuration()['packages'])
        self.assertEqual(
            sorted(package_directories() - listed),
            [],
            'packages that are not in pyproject.toml, and so not in the '
            'wheel',
        )

    def test_every_listed_package_exists(self):
        listed = set(configuration()['packages'])
        self.assertEqual(
            sorted(listed - package_directories()),
            [],
            'packages in pyproject.toml that have no directory',
        )

    def test_the_packages_are_taken_from_the_root_of_the_repository(self):
        self.assertEqual(configuration()['package-dir']['pylorenzmie'], '.')

    def test_every_package_dir_names_a_directory(self):
        for name, folder in configuration()['package-dir'].items():
            with self.subTest(package=name):
                self.assertTrue((ROOT / folder).is_dir())

    def test_the_files_the_program_reads_are_in_the_wheel(self):
        data = configuration()['package-data']
        self.assertIn('*.ui', data['pylorenzmie.lmtool'])
        self.assertIn('*.joblib', data['pylorenzmie.analysis'])
        self.assertTrue((ROOT / 'lmtool' / 'LMTool.ui').is_file())
        self.assertTrue((ROOT / 'analysis' / 'mlp_estimator.joblib').is_file())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
