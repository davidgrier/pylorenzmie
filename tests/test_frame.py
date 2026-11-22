import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pylorenzmie.utilities import imread

from pylorenzmie.analysis import BaseEstimator, DEEstimator, Frame
from pylorenzmie.analysis.Hologram import Hologram


THIS_DIR = Path(__file__).parent
TEST_IMAGE = THIS_DIR / 'data' / 'image0010.png'


class _Fixed(BaseEstimator):
    '''An estimator that is not one of the package.'''

    @BaseEstimator.properties.getter
    def properties(self):
        return {}

    def estimate(self, hologram):
        return pd.Series({'a_p': 3.})


class TestFrame(unittest.TestCase):

    def setUp(self):
        self.frame = Frame()
        self.data = imread(TEST_IMAGE).astype(float) / 100.

    def test_is_hologram(self):
        '''Frame is-a Hologram.'''
        self.assertIsInstance(self.frame, Hologram)

    def test_features_have_the_estimator_that_the_factory_makes(self):
        frame = Frame(estimator=_Fixed)
        frame.data = self.data
        a = frame[0:50, 0:50]
        b = frame[0:50, 50:100]
        self.assertIsInstance(a.estimator, _Fixed)
        self.assertIsNot(a.estimator, b.estimator)
        self.assertTrue(np.array_equal(a.estimator.exclude,
                                       a.mask.exclude))

    def test_the_default_estimator_is_the_global_one(self):
        self.frame.data = self.data
        self.assertIsInstance(self.frame[0:50, 0:50].estimator, DEEstimator)

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

    def test_detect_groups_overlap_into_pair_feature(self):
        '''image0010's two overlapping boxes become one Pair Feature.'''
        from pylorenzmie.analysis import PairEstimator
        from pylorenzmie.theory import Pair
        self.frame.instrument.wavelength = 0.447
        # Wide enough that the two particles' boxes overlap (6 already
        # suffices; the default of 4 keeps them separate).
        self.frame.localizer.nfringes = 10
        self.frame.data = self.data
        n = self.frame.detect()
        self.assertEqual(n, 1)
        feature = self.frame.features[0]
        self.assertIsInstance(feature.particle, Pair)
        self.assertIsInstance(feature.estimator, PairEstimator)
        self.assertEqual(len(feature.centers), 2)
        # shared instrument still reaches the feature's model
        self.assertAlmostEqual(feature.model.instrument.wavelength, 0.447)

    def _pair_frame(self, split_redchi=2.0):
        '''Frame with one box around a noiseless two-sphere hologram.'''
        from pylorenzmie.theory import LorenzMie, Pair
        frame = Frame(split_redchi=split_redchi)
        frame.instrument.wavelength = 0.447
        frame.instrument.magnification = 0.048
        frame.instrument.n_m = 1.34
        truth = LorenzMie(
            particle=Pair(x_p1=85.,
                          y_p1=85.,
                          z_p1=250.,
                          a_p1=0.5,
                          n_p1=1.42,
                          x_p2=115.,
                          y_p2=115.,
                          z_p2=250.,
                          a_p2=0.6,
                          n_p2=1.40),
            instrument=frame.instrument)
        truth.coordinates = LorenzMie.meshgrid((201, 201))
        frame.data = truth.hologram().reshape(201, 201)
        frame.bboxes = ((0, 0), 201, 201)
        # Pin every random choice of the fit: the seed of the DE search, and
        # the seed of the subsample of pixels that the least-squares fit
        # uses, which the pairs that the frame tries share with this
        # feature.  Without them the fit settles in a different minimum from
        # run to run, and the redchi below was sometimes 3.
        frame.features[0].estimator.seed = 0
        frame.features[0].mask.seed = 0
        return frame

    def test_poor_single_fit_splits_into_pair(self):
        '''High-redchi single fit of two spheres is refit as a Pair.'''
        from pylorenzmie.theory import Pair
        frame = self._pair_frame()
        frame.estimate()
        results = frame.optimize()
        self.assertIsInstance(frame.features[0].particle, Pair)
        self.assertLess(results.redchi.iloc[0], 2.0)

    def test_split_disabled_keeps_single(self):
        '''split_redchi=None leaves a single-particle fit alone.'''
        frame = self._pair_frame(split_redchi=None)
        frame.estimate()
        frame.optimize()
        self.assertIsNone(frame.features[0].centers)


class TestFrameOnAnImage(unittest.TestCase):
    '''The pipeline on a real image of two spheres.

    Splitting a feature into a pair is turned off.  It refits both features
    with a differential-evolution search for each of eight candidate pairs,
    which takes a minute on this image and finds no pair; the tests above
    cover the split.
    '''

    def setUp(self):
        self.frame = Frame(split_redchi=None)
        self.data = imread(TEST_IMAGE).astype(float) / 100.

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
