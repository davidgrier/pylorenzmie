import unittest

from analysis import Frame

import os
import cv2
import numpy as np
import pandas as pd
    

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(THIS_DIR, 'data/image0010.png')


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.frame = Frame()
        self.data = cv2.imread(TEST_IMAGE, 0).astype(float)/100.

    def test_data(self):
        '''setting data should set the frame shape 
        and allocate coordinates'''
        self.frame.data = self.data
        self.assertEqual(self.data.size, self.frame.coordinates.size/2)

    def test_detect(self):
        self.frame.data = None
        nfeatures = self.frame.detect()
        self.assertEqual(nfeatures, 0)
        self.frame.data = self.data
        nfeatures = self.frame.detect()
        features = self.frame.features
        bboxes = self.frame.bboxes
        self.assertEqual(nfeatures, len(features))
        self.assertEqual(nfeatures, len(bboxes))

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
        self.assertEqual(len(self.frame.results), 2)

    def test_shape(self):
        shape = [640, 480]
        self.frame.shape = None
        self.frame.shape = shape
        self.frame.shape = None
        self.frame.shape = shape
        self.frame.shape = shape
        self.assertListEqual(self.frame.shape, shape)

if __name__ == '__main__':
    unittest.main()
