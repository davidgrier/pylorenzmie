'''BaseEstimator.find_center on simulated holograms.'''

import unittest

import numpy as np

from pylorenzmie.analysis import DEEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie

SHAPE = (151, 101)               # a crop that is not square
CORNER = (200., 300.)
TRUTH = (230., 360.)             # well off the middle of the crop


class TestFindCenter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model = LorenzMie()
        model.coordinates = model.meshgrid(SHAPE, corner=CORNER)
        x_p, y_p = TRUTH
        sphere = dict(z_p=150., a_p=0.8, n_p=1.45)
        model.properties = dict(sphere, x_p=x_p, y_p=y_p)
        data = model.hologram().reshape(SHAPE).copy()
        data += np.random.default_rng(0).normal(0., 0.02, SHAPE)
        cls.hologram = Hologram(data, CORNER)

    def find(self, center=None, max_shift=None, exclude=None):
        estimator = DEEstimator(model=LorenzMie(), exclude=exclude)
        estimator.center = center
        estimator.max_shift = max_shift
        return np.array(estimator.find_center(self.hologram))

    def assertNear(self, found, truth, delta):
        self.assertLess(np.hypot(*(found - truth)), delta)

    def test_a_hint_is_refined(self):
        found = self.find(center=(TRUTH[0] + 3.5, TRUTH[1] - 4.))
        self.assertNear(found, TRUTH, 0.6)

    def test_the_search_stays_within_its_range(self):
        center = (TRUTH[0] + 12., TRUTH[1])
        found = self.find(center=center)
        self.assertTrue(np.all(np.abs(found - center) <= 5.))

    def test_a_wider_range_can_be_asked_for(self):
        found = self.find(center=(TRUTH[0] + 9., TRUTH[1]), max_shift=12.)
        self.assertNear(found, TRUTH, 0.6)

    def test_without_a_hint_the_middle_is_kept(self):
        middle = (250., 375.)
        np.testing.assert_allclose(self.find(), middle)

    def test_no_search_keeps_the_hint(self):
        center = (233., 358.)
        np.testing.assert_allclose(self.find(center=center, max_shift=0.),
                                   center)

    def test_excluded_pixels_are_not_used(self):
        exclude = np.ones(SHAPE, bool)
        center = (233., 358.)
        found = self.find(center=center, exclude=exclude)
        np.testing.assert_allclose(found, center)

    def test_data_that_cannot_tell_keep_the_hint(self):
        estimator = DEEstimator(model=LorenzMie())
        estimator.center = (20., 20.)
        found = estimator.find_center(Hologram(np.ones((41, 41))))
        np.testing.assert_allclose(found, (20., 20.))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
