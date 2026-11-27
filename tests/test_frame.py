import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pylorenzmie.utilities import imread

from pylorenzmie.analysis import (BaseEstimator, DEEstimator, Frame,
                                  Optimizer)
from pylorenzmie.analysis.Frame import TRIAL_NFEV
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

    def test_features_leave_out_the_saturated_pixels(self):
        saturated = self.data >= 2.55
        self.frame.data = self.data
        self.frame.saturated = saturated
        feature = self.frame[600:800, 500:650]
        self.assertTrue(feature.saturated.any())
        np.testing.assert_array_equal(feature.mask.exclude,
                                      saturated[600:800, 500:650])

    def test_new_data_has_no_saturated_pixels(self):
        frame = Frame(self.data, saturated=self.data >= 2.55)
        self.assertTrue(frame.saturated.any())
        frame.data = self.data
        self.assertIsNone(frame.saturated)

    def test_analyze_takes_the_saturated_pixels(self):
        frame = Frame()
        saturated = np.zeros((64, 64), dtype=bool)
        frame.analyze(np.ones((64, 64)), saturated)
        self.assertIs(frame.saturated, saturated)

    def test_analyze_without_data_keeps_the_data(self):
        data = np.ones((64, 64))
        self.frame.data = data
        self.frame.analyze()
        self.assertIs(self.frame.data, data)

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

    def test_overlapping_boxes_are_single_particles(self):
        self.frame.localizer.nfringes = 10     # as below: the boxes overlap
        self.frame.data = self.data
        self.assertEqual(self.frame.detect(), 2)
        for feature in self.frame.features:
            self.assertEqual(len(feature.centers), 1)
            self.assertEqual(feature.estimator.center, feature.centers[0])

    def test_a_box_that_crosses_the_edge_is_trimmed(self):
        bbox = [((-30, 0), 100, 100)]
        found = pd.DataFrame(dict(x_p=[20.], y_p=[50.], bbox=bbox))
        frame = Frame(self.data, estimator=_Fixed)
        with patch.object(frame.localizer, 'localize', return_value=found):
            frame.detect()
        feature = frame.features[0]
        self.assertEqual(feature.corner, (0., 0.))
        self.assertEqual(feature.shape, (100, 70))
        self.assertEqual(feature.centers, [(20., 50.)])
        self.assertIsInstance(feature.estimator, _Fixed)
        self.assertEqual(feature.estimator.center, (20., 50.))

    def test_a_given_box_is_not_made_square(self):
        frame = Frame(self.data)
        frame.bboxes = ((100, 200), 80, 40)
        feature = frame.features[0]
        self.assertEqual(feature.corner, (100., 200.))
        self.assertEqual(feature.shape, (40, 80))
        self.assertIsNone(feature.centers)

    def test_a_single_feature_with_a_center_can_be_split(self):
        frame = self._pair_frame()
        frame.features[0].estimator.center = None
        feature = frame._feature((slice(0, 201), slice(0, 201)),
                                 [(100., 100.)])
        frame._features = [feature]
        result = pd.Series(dict(redchi=5.))
        with patch.object(frame, '_split', return_value=feature) as split:
            with patch('pylorenzmie.analysis.Frame._fit',
                       return_value=result):
                frame.split_suspects()
        split.assert_called_once_with(feature)

    def test_no_detection_is_dropped(self):
        found = pd.DataFrame(dict(
            x_p=[60., 90., 120.],
            y_p=[60., 60., 60.],
            bbox=[((x - 30, 30), 60, 60) for x in (60, 90, 120)]))
        for pairs in (False, True):
            with self.subTest(pairs=pairs):
                frame = Frame(self.data, pairs=pairs)
                localizer = frame.localizer
                with patch.object(localizer, 'localize', return_value=found):
                    self.assertEqual(frame.detect(), 3)

    def test_pairs_are_something_to_ask_for(self):
        frame = Frame()
        self.assertFalse(frame.pairs)
        self.assertIsNone(frame.split_redchi)

    def test_detect_groups_overlap_into_pair_feature(self):
        '''image0010's two overlapping boxes become one Pair Feature.'''
        from pylorenzmie.analysis import PairEstimator
        from pylorenzmie.theory import Pair
        self.frame.pairs = True
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
        limits = []
        optimize = Optimizer.optimize

        def recorded(optimizer, hologram):
            limits.append((len(optimizer.variables),
                           optimizer.settings['max_nfev']))
            return optimize(optimizer, hologram)

        with patch.object(Optimizer, 'optimize', recorded):
            results = frame.optimize()
        self.assertIsInstance(frame.features[0].particle, Pair)
        self.assertLess(results.redchi.iloc[0], 2.0)
        # the single fit has the usual budget; each trial of a pair less,
        # and the pair that is chosen has the usual one again
        self.assertEqual(limits, [(5, 2000)] + 8 * [(10, TRIAL_NFEV)])
        chosen = frame.features[0].optimizer
        self.assertEqual(chosen.settings['max_nfev'], 2000)

    def test_split_disabled_keeps_single(self):
        '''split_redchi=None leaves a single-particle fit alone.'''
        frame = self._pair_frame(split_redchi=None)
        frame.estimate()
        frame.optimize()
        self.assertIsNone(frame.features[0].centers)


class TestFrameOnAnImage(unittest.TestCase):
    '''The pipeline on a real image of two spheres.'''

    def setUp(self):
        self.frame = Frame()
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
