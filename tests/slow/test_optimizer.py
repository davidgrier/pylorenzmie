import unittest

from pylorenzmie.analysis import Optimizer
from pylorenzmie.theory import LorenzMie
from pylorenzmie.lib import LMObject
from pathlib import Path
import cv2
import numpy as np


THIS_DIR = Path(__file__).parent.parent
TEST_IMAGE = str(THIS_DIR / 'data' / 'crop.png')


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE).astype(float)
        img /= np.mean(img)
        img = img[::4, ::4]
        shape = img.shape
        coords = 4. * LMObject.meshgrid(shape)
        model = LorenzMie(coordinates=coords)
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [shape[0] // 2, shape[1] // 2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.data = img.ravel()
        self.optimizer = Optimizer(model=model)

    def test_optimize(self):
        self.optimizer.data = self.data
        result = self.optimizer.optimize()
        self.assertTrue(result.success)

    def test_report_after_optimize(self):
        '''report() covers exactly the fitted variables after optimize()'''
        self.optimizer.data = self.data
        self.optimizer.optimize()
        report = self.optimizer.report()
        for v in self.optimizer.variables:
            self.assertIn(v, report)
        self.assertIn('χ²', report)

    def test_optimize_failure(self):
        self.optimizer.data = self.data + 100.
        result = self.optimizer.optimize()
        failure = not result.success or (result.redchi > 100.)
        self.assertTrue(failure)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
