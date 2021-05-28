import unittest

from fitting import Optimizer
from theory import LMHologram
from utilities import coordinates
import os
import cv2
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(THIS_DIR, 'data/crop.png')


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        img = cv2.imread(TEST_IMAGE, 0).astype(float)
        img /= np.mean(img)
        img = img[::4,::4]
        self.shape = img.shape
        self.data = img.ravel()
        self.coordinates = 4.*coordinates(self.shape)
        model = LMHologram(coordinates=self.coordinates)
        model.instrument.wavelength = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m = 1.34
        model.particle.r_p = [self.shape[0]//2, self.shape[1]//2, 330]
        model.particle.a_p = 1.1
        model.particle.n_p = 1.4
        self.optimizer = Optimizer(model=model)

    def test_report_none(self):
        r = self.optimizer.report
        self.assertIs(r, None)

    def test_data(self):
        self.optimizer.data = self.data
        self.assertEqual(self.optimizer.data.size, self.data.size)

    def test_coordinates(self):
        self.optimizer.coordinates = None
        self.assertIs(self.optimizer.coordinates, None)
        self.optimizer.coordinates = self.coordinates
        self.assertEqual(self.optimizer.coordinates.shape[1],
                         self.coordinates.shape[1])

    def test_optimize(self, method='lm'):
        self.optimizer.method = method
        self.optimizer.data = self.data
        result = self.optimizer.optimize()
        if not result.success:
            print(result)
        self.assertTrue(result.success)

    def test_optimize_failure(self):
        self.optimizer.data = self.data + 100.
        result = self.optimizer.optimize()
        failure = not result.success or (result.redchi > 100.)
        self.assertTrue(failure)

    def test_metadata(self):
        self.assertIsInstance(self.optimizer.metadata, pd.Series)

    def test_properties(self):
        properties = self.optimizer.properties
        self.optimizer.properties = properties
        self.assertTrue('settings' in properties)

        
if __name__ == '__main__':
    unittest.main()
