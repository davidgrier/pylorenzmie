'''CATCHAnalyzer, with a stand-in for CATCH, and with CATCH where it is.'''

import importlib
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from pylorenzmie.analysis import (BaseEstimator, CATCHAnalyzer, DEEstimator,
                                  Feature, Frame, catch_available)
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import Instrument, LorenzMie, Pair

# the package namespace holds the class of the same name
analyzer = importlib.import_module('pylorenzmie.analysis.CATCHAnalyzer')

OPTICS = dict(wavelength=0.447, magnification=0.048, n_m=1.34)
SHAPE = (201, 201)
TRUTH = dict(x_p=100.3, y_p=99.6, z_p=150., a_p=0.75, n_p=1.45)


def detections(confidences, **estimate):
    '''A table as CATCH reports it, one row for each confidence.'''
    values = dict(z_p=150., a_p=0.75, n_p=1.45) | estimate
    rows = []
    for i, confidence in enumerate(confidences):
        box = ((20 + i, 20), 160, 160)
        detection = dict(x_p=100. + i, y_p=100., bbox=box)
        report = dict(confidence=confidence, quality='')
        rows.append(detection | report | values)
    return pd.DataFrame(rows)


class StandIn:
    '''CATCHapp, for an instrument whose pixels are *scale* times those of
    its reference.'''

    def __init__(self, found=None, scale=1., estimates=None):
        self.found = pd.DataFrame() if found is None else found
        self.scale = scale
        self.estimates = estimates
        self.localizer = mock.Mock(scales=(1., 2.5))
        self.estimator = mock.Mock(scales=(1., 2.5),
                                   side_effect=self._estimate)
        self.instrument = None
        self.saturation = None

    def analyze(self, image, saturation=None):
        self.saturation = saturation
        return self.found.copy()

    def _estimate(self, image, x, y, instrument, saturation=None):
        self.saturation = saturation
        self.position = (x[0], y[0])
        if self.estimates is not None:
            return self.estimates
        return pd.DataFrame(dict(z_p=[150.], a_p=[0.75], n_p=[1.45]))


class Stubbed(unittest.TestCase):

    def stand_in(self, **kwargs):
        app = StandIn(**kwargs)
        patcher = mock.patch.object(analyzer, '_app', return_value=app)
        patcher.start()
        self.addCleanup(patcher.stop)
        return app

    def hologram(self):
        model = LorenzMie(coordinates=LorenzMie.meshgrid(SHAPE))
        model.instrument.properties = OPTICS
        model.properties = TRUTH
        noise = np.random.default_rng(0).normal(0., 0.02, SHAPE)
        return model.hologram().reshape(SHAPE) + noise


class TestAnalyzer(Stubbed):

    def test_it_is_an_estimator(self):
        self.assertIsInstance(CATCHAnalyzer(), BaseEstimator)
        self.assertEqual(CATCHAnalyzer().properties, dict(compile=False))

    def test_localize_gives_catch_the_optics_and_the_saturation(self):
        app = self.stand_in(found=detections([0.9]))
        catch = CATCHAnalyzer(Instrument(**OPTICS))
        catch.saturated = np.zeros((64, 64), bool)
        found = catch.localize(np.ones((64, 64)))
        self.assertEqual(app.instrument, OPTICS)
        self.assertIs(app.saturation, catch.saturated)
        self.assertEqual(found.attrs['nfound'], 1)
        self.assertEqual(found.attrs['warning'], '')

    def test_no_particles_is_a_table_with_the_columns(self):
        self.stand_in()
        found = CATCHAnalyzer().localize(np.ones((64, 64)))
        self.assertEqual(len(found), 0)
        for column in ('x_p', 'y_p', 'bbox', 'z_p', 'quality'):
            self.assertIn(column, found.columns)

    def test_pixels_outside_the_range_of_the_networks_are_reported(self):
        for scale, warned in ((0.5, True), (1., False), (2.504, False),
                              (2.6, False), (3., True)):
            with self.subTest(scale=scale):
                app = StandIn(scale=scale)
                with mock.patch.object(analyzer, '_app', return_value=app):
                    found = CATCHAnalyzer().localize(np.ones((64, 64)))
                self.assertEqual(bool(found.attrs['warning']), warned)

    def test_estimate_of_a_crop_at_the_center_of_a_feature(self):
        app = self.stand_in()
        data = self.hologram()
        catch = CATCHAnalyzer(Instrument(**OPTICS))
        feature = Feature(Hologram(data, (40., 60.)),
                          centers=[(140., 160.)],
                          estimator=catch)
        result = feature.estimate()
        self.assertEqual(list(result.index),
                         ['x_p', 'y_p', 'z_p', 'a_p', 'n_p'])
        self.assertAlmostEqual(result.x_p, 140.3, delta=0.6)
        np.testing.assert_allclose(app.position,
                                   (result.x_p - 40., result.y_p - 60.))
        self.assertEqual(feature.particle.a_p, 0.75)
        self.assertEqual(app.saturation.shape, data.shape)

    def test_no_estimate_is_an_error(self):
        nothing = pd.DataFrame(np.nan,
                               index=[0],
                               columns=['z_p', 'a_p', 'n_p'])
        self.stand_in(estimates=nothing)
        with self.assertRaisesRegex(ValueError, 'no estimate'):
            CATCHAnalyzer().estimate(Hologram(np.ones((64, 64))))


class TestFrameWithCATCH(Stubbed):

    def frame(self, found, **kwargs):
        self.stand_in(found=found)
        catch = CATCHAnalyzer(Instrument(**OPTICS))
        return Frame(self.hologram(), localizer=catch, **kwargs)

    def found(self, **estimate):
        values = {p: TRUTH[p] for p in ('z_p', 'a_p', 'n_p')}
        table = detections([0.9], **(values | estimate))
        table['x_p'], table['y_p'] = TRUTH['x_p'], TRUTH['y_p']
        return table

    def test_the_frame_shares_the_instrument_of_the_analyzer(self):
        frame = self.frame(self.found())
        self.assertIs(frame.instrument, frame.localizer.instrument)

    def test_fits_start_from_the_estimates_without_a_search(self):
        start = self.found(z_p=152., a_p=0.76, n_p=1.46)
        frame = self.frame(start)
        with mock.patch.object(DEEstimator, 'estimate') as search:
            frame.detect()
            frame.estimate()
        search.assert_not_called()
        particle = frame.features[0].particle
        self.assertEqual((particle.x_p, particle.z_p, particle.a_p),
                         (TRUTH['x_p'], 152., 0.76))
        results = frame.optimize()
        self.assertAlmostEqual(results.a_p.iloc[0], TRUTH['a_p'], delta=0.01)
        self.assertEqual(list(results.confidence), [0.9])
        self.assertEqual(list(results.quality), [''])

    def test_fits_are_robust_unless_asked_otherwise(self):
        for robust, expected in ((None, True), (True, True), (False, False)):
            with self.subTest(robust=robust):
                frame = self.frame(self.found(), robust=robust)
                frame.detect()
                optimizer = frame.features[0].optimizer
                self.assertEqual(optimizer.robust, expected)

    def test_splitting_keeps_least_squares(self):
        frame = self.frame(self.found(), split_redchi=2.)
        frame.detect()
        self.assertFalse(frame.features[0].optimizer.robust)

    def test_splitting_and_robust_fits_are_not_combined(self):
        frame = self.frame(self.found(), split_redchi=2., robust=True)
        with self.assertRaisesRegex(ValueError, 'split_redchi'):
            frame.detect()

    def test_an_estimator_that_is_given_is_used_instead(self):
        frame = self.frame(self.found(), estimator=lambda: DEEstimator(
            model=LorenzMie(), seed=0))
        frame.detect()
        self.assertIsInstance(frame.features[0].estimator, DEEstimator)

    def test_a_particle_without_an_estimate_keeps_its_row(self):
        found = pd.concat([self.found(),
                           self.found(z_p=np.nan).assign(quality='low snr')],
                          ignore_index=True)
        frame = self.frame(found, split_redchi=2.)
        results = frame.analyze()
        self.assertEqual(len(results), 2)
        self.assertEqual(list(results.success), [True, False])
        self.assertEqual(results.x_p.iloc[1], TRUTH['x_p'])
        self.assertTrue(np.isnan(results.a_p.iloc[1]))
        self.assertEqual(results.npix.iloc[1], 0)
        self.assertEqual(results.quality.iloc[1], 'low snr')

    def test_saturated_pixels_go_to_the_analyzer(self):
        frame = self.frame(self.found())
        saturated = np.zeros(SHAPE, bool)
        frame.saturated = saturated
        frame.detect()
        self.assertIs(frame.localizer.saturated, saturated)

    def test_other_optics_are_refused(self):
        frame = self.frame(self.found(), instrument=Instrument(wavelength=0.6))
        with self.assertRaisesRegex(ValueError, 'wavelength'):
            frame.detect()

    def test_pairs_are_fitted_as_pairs_and_singles_keep_their_estimates(self):
        found = detections([0.9, 0.8, 0.7])
        found['bbox'] = [((0, 0), 60, 60), ((40, 0), 60, 60),
                         ((130, 130), 60, 60)]
        frame = self.frame(found, pairs=True)
        self.assertEqual(frame.detect(), 2)
        pair, single = frame.features
        self.assertIsInstance(pair.particle, Pair)
        single.estimate()
        self.assertEqual(single.particle.z_p, 150.)


@unittest.skipUnless(catch_available(), 'CATCH is not installed')
class TestCATCH(unittest.TestCase):
    '''The real networks, on a simulated hologram.'''

    @classmethod
    def setUpClass(cls):
        shape = (400, 400)
        model = LorenzMie(coordinates=LorenzMie.meshgrid(shape))
        model.instrument.properties = OPTICS
        sphere = dict(z_p=300., a_p=1., n_p=1.45)
        model.properties = dict(sphere, x_p=190., y_p=210.)
        noise = np.random.default_rng(0).normal(0., 0.02, shape)
        cls.image = model.hologram().reshape(shape) + noise

    def test_a_frame_finds_and_fits_the_particle(self):
        catch = CATCHAnalyzer(Instrument(**OPTICS))
        results = Frame(localizer=catch).analyze(self.image)
        self.assertEqual(len(results), 1)
        fit = results.iloc[0]
        for name, truth, delta in (('x_p', 190., 0.5), ('y_p', 210., 0.5),
                                   ('z_p', 300., 3.), ('a_p', 1., 0.02),
                                   ('n_p', 1.45, 0.01)):
            self.assertAlmostEqual(fit[name], truth, delta=delta)
        self.assertEqual(fit.quality, '')

    def test_a_crop_is_estimated(self):
        catch = CATCHAnalyzer(Instrument(**OPTICS))
        crop = Hologram(self.image[100:300, 100:300], (100., 100.))
        feature = Feature(crop, centers=[(190., 210.)], estimator=catch)
        result = feature.estimate()
        self.assertAlmostEqual(result.a_p, 1., delta=0.15)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
