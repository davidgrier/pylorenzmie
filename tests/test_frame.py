import unittest
import numpy as np
from pathlib import Path
import cv2

from pylorenzmie.analysis import Frame
from pylorenzmie.analysis.Hologram import Hologram


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'image0010.png'


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.frame = Frame()
        self.data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE).astype(float) / 100.

    def test_is_hologram(self):
        '''Frame is-a Hologram.'''
        self.assertIsInstance(self.frame, Hologram)

    def test_data(self):
        '''Setting data allocates a coordinate grid matching the image size.'''
        self.frame.data = self.data
        self.assertEqual(self.data.size, self.frame.coordinates.size // 2)

    def test_shape_from_data(self):
        '''shape reflects the most recently assigned image dimensions.'''
        self.frame.data = self.data
        self.assertEqual(self.frame.shape, self.data.shape)

    def test_shape_no_data(self):
        '''shape is (0, 0) when no data has been assigned.'''
        self.assertEqual(self.frame.shape, (0, 0))

    def test_bboxes(self):
        self.frame.data = self.data
        self.frame.bboxes = ((100, 200), 300, 400)
        self.assertEqual(len(self.frame.features), 1)

    def test_bboxes_propagate_instrument(self):
        self.frame.instrument.wavelength = 0.447
        self.frame.data = self.data
        self.frame.bboxes = ((100, 200), 100, 100)
        self.assertAlmostEqual(
            self.frame.features[0].model.instrument.wavelength, 0.447)

    def test_detect_no_data(self):
        self.frame.data = None
        self.assertEqual(self.frame.detect(), 0)

    def test_detect(self):
        self.frame.data = self.data
        nfeatures = self.frame.detect()
        self.assertEqual(nfeatures, len(self.frame.features))
        self.assertEqual(nfeatures, len(self.frame.bboxes))

    def test_detect_propagates_instrument(self):
        '''Instrument settings must reach each Feature's model.'''
        self.frame.instrument.wavelength = 0.447
        self.frame.data = self.data
        self.frame.detect()
        if len(self.frame.features) > 0:
            self.assertAlmostEqual(
                self.frame.features[0].model.instrument.wavelength, 0.447)

    def test_getitem_returns_feature(self):
        '''Slicing a Frame returns a Feature.'''
        from pylorenzmie.analysis import Feature
        self.frame.data = self.data
        crop = self.frame[50:150, 50:150]
        self.assertIsInstance(crop, Feature)

    def test_getitem_corner(self):
        '''Sliced Feature has corner matching the slice origin.'''
        self.frame.data = self.data
        crop = self.frame[50:150, 50:150]
        self.assertEqual(crop.corner, (50., 50.))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
