import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from pylorenzmie.analysis.Localizer import Localizer


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'image0010.png'


class TestLocalizer(unittest.TestCase):

    def setUp(self):
        import cv2
        self.localizer = Localizer()
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE)
        self.data = data.astype(float) / 100.

    def test_properties(self):
        props = self.localizer.properties
        self.assertIn('diameter', props)
        self.assertIn('nfringes', props)
        self.assertEqual(props['diameter'], 31)
        self.assertEqual(props['nfringes'], 20)

    def test_json_roundtrip(self):
        self.localizer.diameter = 41
        s = self.localizer.to_json()
        self.localizer.diameter = 31
        self.localizer.from_json(s)
        self.assertEqual(self.localizer.diameter, 41)

    def test_detect_alias(self):
        '''detect() is a backward-compatibility alias for localize()'''
        self.assertEqual(self.localizer.detect, self.localizer.localize)

    def test_localize_returns_dataframe(self):
        result = self.localizer.localize(self.data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_localize_columns(self):
        result = self.localizer.localize(self.data)
        self.assertIn('x_p', result.columns)
        self.assertIn('y_p', result.columns)
        self.assertIn('bbox', result.columns)

    def test_localize_finds_features(self):
        result = self.localizer.localize(self.data)
        self.assertGreater(len(result), 0)

    def test_localize_list(self):
        result = self.localizer.localize([self.data, self.data])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_crop(self):
        result = self.localizer.localize(self.data)
        if len(result) > 0:
            bbox = result.iloc[0]['bbox']
            cropped = self.localizer.crop(self.data, bbox)
            self.assertIsInstance(cropped, np.ndarray)
            self.assertEqual(cropped.ndim, 2)

    def test_crop_shape(self):
        bbox = ((10, 20), 50, 60)
        cropped = self.localizer.crop(self.data, bbox)
        self.assertEqual(cropped.shape, (60, 50))

    def test_custom_diameter(self):
        result = self.localizer.localize(self.data, diameter=41)
        self.assertIsInstance(result, pd.DataFrame)

    def test_custom_nfringes(self):
        result = self.localizer.localize(self.data, nfringes=15)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
