import unittest
import numpy as np
from pathlib import Path
import cv2

from pylorenzmie.analysis import Frame


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'image0010.png'


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.frame = Frame()
        self.data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE).astype(float) / 100.

    def test_data(self):
        '''Setting data allocates a coordinate grid matching the image size.'''
        self.frame.data = self.data
        self.assertEqual(self.data.size, self.frame.coordinates.size // 2)

    def test_shape_stored_as_tuple(self):
        self.frame.shape = [640, 480]
        self.assertIsInstance(self.frame.shape, tuple)
        self.assertEqual(self.frame.shape, (640, 480))

    def test_shape_unchanged_on_repeat(self):
        self.frame.shape = (640, 480)
        coords_id = id(self.frame.coordinates)
        self.frame.shape = (640, 480)
        self.assertEqual(id(self.frame.coordinates), coords_id)

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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
