import unittest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from pylorenzmie.analysis import Estimator, Hologram, BaseEstimator
from pylorenzmie.lib import LMObject
from pylorenzmie.theory import Instrument, LorenzMie


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'crop.png'


class TestEstimator(unittest.TestCase):

    def setUp(self):
        self.instrument = Instrument()
        self.estimator = Estimator(instrument=self.instrument)
        data = cv2.imread(str(TEST_IMAGE), cv2.IMREAD_GRAYSCALE)
        self.hologram = Hologram(data.astype(float) / 100.)

    def test_is_base_estimator(self):
        '''Estimator is a subclass of BaseEstimator.'''
        self.assertIsInstance(self.estimator, BaseEstimator)

    def test_properties_keys(self):
        '''properties contains all particle parameter keys.'''
        self.estimator.estimate(self.hologram)
        for key in ('x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p'):
            self.assertIn(key, self.estimator.properties)

    def test_estimate_returns_series(self):
        '''estimate returns a pandas.Series.'''
        result = self.estimator.estimate(self.hologram)
        self.assertIsInstance(result, pd.Series)

    def test_center_of_the_crop_is_kept_without_a_search(self):
        '''max_shift = 0 puts the particle at the center of the crop.'''
        estimator = Estimator(instrument=self.instrument, max_shift=0.)
        result = estimator.estimate(self.hologram)
        self.assertAlmostEqual(result['x_p'],
                               float(self.hologram.coordinates[0].mean()))
        self.assertAlmostEqual(result['y_p'],
                               float(self.hologram.coordinates[1].mean()))

    def test_center_without_a_search_is_corner_aware(self):
        '''The center shifts with the corner of the hologram.'''
        estimator = Estimator(instrument=self.instrument, max_shift=0.)
        h = Hologram(self.hologram.data, corner=(10., 20.))
        result = estimator.estimate(h)
        self.assertAlmostEqual(result['x_p'], float(h.coordinates[0].mean()))
        self.assertAlmostEqual(result['y_p'], float(h.coordinates[1].mean()))

    def test_search_is_corner_aware(self):
        '''The search reports pixel coordinates of the image, not the crop.'''
        h = Hologram(self.hologram.data, corner=(10., 20.))
        a = self.estimator.estimate(self.hologram)
        b = self.estimator.estimate(h)
        self.assertAlmostEqual(b['x_p'] - a['x_p'], 10.)
        self.assertAlmostEqual(b['y_p'] - a['y_p'], 20.)

    def test_estimate_z_positive(self):
        '''Estimated z_p is positive.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['z_p'], 0)

    def test_estimate_a_positive(self):
        '''Estimated a_p is positive.'''
        result = self.estimator.estimate(self.hologram)
        self.assertGreater(result['a_p'], 0)

    def test_predict_alias(self):
        '''predict() is a backward-compatibility alias for estimate().'''
        r1 = self.estimator.estimate(self.hologram)
        r2 = self.estimator.predict(self.hologram)
        self.assertAlmostEqual(r1['z_p'], r2['z_p'], places=3)


class TestEstimatorOnSyntheticHolograms(unittest.TestCase):
    '''The center, height and radius of a sphere from its hologram.'''

    SHAPE = (301, 301)

    def hologram(self, magnification, a_p, z_um, shift=(0., 0.), n_p=1.6):
        instrument = Instrument()
        instrument.wavelength = 0.447
        instrument.magnification = magnification
        instrument.n_m = 1.34
        model = LorenzMie(instrument=instrument)
        model.coordinates = LMObject.meshgrid(self.SHAPE)
        ny, nx = self.SHAPE
        self.truth = [nx // 2 + shift[0], ny // 2 + shift[1],
                      z_um / magnification]
        model.particle.r_p = self.truth
        model.particle.a_p = a_p
        model.particle.n_p = n_p
        return instrument, Hologram(model.hologram().reshape(self.SHAPE))

    def test_center_is_found_when_the_particle_is_off_center(self):
        '''A particle 8 pixels from the center of the crop is located.'''
        instrument, h = self.hologram(0.12, 0.5, 100., shift=(6., -5.))
        result = Estimator(instrument=instrument, n_p=1.6).estimate(h)
        self.assertAlmostEqual(result['x_p'], self.truth[0], delta=0.5)
        self.assertAlmostEqual(result['y_p'], self.truth[1], delta=0.5)

    def test_radius_does_not_depend_on_the_error_in_the_center(self):
        '''A particle off the center of the crop has the same radius.'''
        radii = []
        for shift in ((0., 0.), (7., 0.), (-5., 5.)):
            instrument, h = self.hologram(0.12, 0.5, 100., shift)
            result = Estimator(instrument=instrument, n_p=1.6).estimate(h)
            radii.append(result['a_p'])
        self.assertLess(max(radii) / min(radii), 1.05)

    def test_radius_and_height_at_any_magnification(self):
        for magnification, a_p, z_um in ((0.048, 0.75, 10.),
                                         (0.12, 0.5, 100.),
                                         (0.12, 1.5, 140.)):
            with self.subTest(magnification=magnification, a_p=a_p):
                instrument, h = self.hologram(magnification, a_p, z_um)
                result = Estimator(instrument=instrument, n_p=1.6).estimate(h)
                self.assertAlmostEqual(result['a_p'] / a_p, 1., delta=0.1)
                self.assertAlmostEqual(result['z_p'] / self.truth[2], 1.,
                                       delta=0.03)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
