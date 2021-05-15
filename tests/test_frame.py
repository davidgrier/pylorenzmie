import unittest

from analysis import Frame

import os
import cv2
import numpy as np

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
        self.frame.data = self.data
        nfeatures = self.frame.detect()
        self.assertEqual(nfeatures, 2)
        

if __name__ == '__main__':
    unittest.main()
