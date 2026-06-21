import unittest

from pylorenzmie.analysis import Frame
from pathlib import Path
import cv2
import pandas as pd


THIS_DIR = Path(__file__).parent.parent
TEST_IMAGE = str(THIS_DIR / 'data' / 'image0010.png')


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.frame = Frame()
        self.data = cv2.imread(TEST_IMAGE, 0).astype(float) / 100.

    def test_optimize(self):
        self.frame.data = self.data
        self.frame.detect()
        particle = self.frame.features[0].particle
        a_p = particle.a_p
        self.frame.estimate()
        self.frame.optimize()
        self.assertNotEqual(particle.a_p, a_p)

    def test_analyze(self):
        results = self.frame.analyze(self.data)
        self.assertIsInstance(results, pd.DataFrame)

    def test_results(self):
        self.frame.analyze(None)
        self.assertEqual(len(self.frame.results), 0)
        self.frame.analyze(self.data)
        self.assertGreater(len(self.frame.results), 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
