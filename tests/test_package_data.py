import importlib
import unittest
from pathlib import Path


class TestPackageData(unittest.TestCase):
    '''Files that must be installed with the package.'''

    def test_the_package_is_marked_as_typed(self):
        # PEP 561: type checkers use the annotations of a package that
        # contains py.typed
        package = importlib.import_module('pylorenzmie')
        self.assertTrue((Path(package.__file__).parent / 'py.typed').exists())

    def test_the_trained_estimator_is_installed(self):
        analysis = importlib.import_module('pylorenzmie.analysis')
        weights = Path(analysis.__file__).parent / 'mlp_estimator.joblib'
        self.assertTrue(weights.exists())

    def test_the_gui_files_are_installed(self):
        lmtool = importlib.import_module('pylorenzmie.lmtool')
        folder = Path(list(lmtool.__path__)[0])    # has no __init__.py
        for name in ('LMTool.ui', 'LMTool.json', 'ParameterWidget.ui'):
            with self.subTest(name):
                self.assertTrue((folder / name).exists())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
