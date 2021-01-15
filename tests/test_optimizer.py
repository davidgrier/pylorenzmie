import unittest

from fitting import Optimizer
from theory import (LMHologram, coordinates)
import os
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(THIS_DIR, 'data/crop.png')


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        img = cv2.imread(TEST_IMAGE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
        img /= np.mean(img)
        self.shape = img.shape
        self.data = img.flatten()
        model = LMHologram(coordinates=coordinates(self.shape))
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [self.shape[0]//2, self.shape[1]//2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.optimizer = Optimizer(model=model)

    def test_report_none(self):
        r = self.optimizer.report()
        self.assertIs(r, None)

    def test_data(self):
        self.optimizer.data = self.data
        self.assertEqual(self.optimizer.data.size, self.data.size)

    def test_optimize(self, method='lm', robust=False):
        self.optimizer.method = method
        self.optimizer.data = self.data
        result = self.optimizer.optimize(robust=robust)
        if not result.success:
            print(result)
        self.assertTrue(result.success)

    def test_optimize_amoeba(self):
        self.test_optimize(method='amoeba')

    def test_optimize_amoeba_robust(self):
        self.test_optimize(method='amoeba', robust=True)

    def test_optimize_lm_amoeba(self):
        self.test_optimize(method='amoeba-lm')

    def test_optimize_failure(self):
        self.optimizer.method = 'lm'
        self.optimizer.data = self.data + 100.
        result = self.optimizer.optimize()
        failure = not result.success or (result.redchi > 100.)
        self.assertTrue(failure)

    def test_dumps_loads(self):
        s = self.optimizer.dumps()
        self.optimizer.loads(s)
        self.assertIsInstance(s, str)


if __name__ == '__main__':
    unittest.main()
