import unittest

import numpy as np

from pylorenzmie.theory import LorenzMie
from pylorenzmie.utilities import estimate_noise


class TestEstimateNoise(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        model = LorenzMie(coordinates=LorenzMie.meshgrid((300, 300)))
        model.particle.r_p = [150., 150., 200.]
        self.hologram = model.hologram().reshape(300, 300)

    def test_white_noise(self):
        for sigma in (0.01, 0.05, 0.2):
            with self.subTest(sigma=sigma):
                image = 1. + self.rng.normal(0., sigma, (300, 300))
                self.assertAlmostEqual(estimate_noise(image) / sigma, 1.,
                                       delta=0.03)

    def test_the_fringes_of_a_hologram_are_not_noise(self):
        self.assertLess(estimate_noise(self.hologram), 1e-3)
        sigma = 0.05
        noisy = self.hologram + self.rng.normal(0., sigma, (300, 300))
        self.assertAlmostEqual(estimate_noise(noisy) / sigma, 1., delta=0.03)

    def test_a_small_bright_feature_does_not_change_it(self):
        image = 1. + self.rng.normal(0., 0.05, (300, 300))
        reference = estimate_noise(image)
        image[100:110, 100:110] += 5.
        self.assertAlmostEqual(estimate_noise(image) / reference, 1.,
                               delta=0.02)

    def test_a_gradient_is_not_noise(self):
        ramp = np.linspace(0.5, 1.5, 200)[None, :] * np.ones((200, 1))
        self.assertLess(estimate_noise(ramp), 1e-9)

    def test_scales_with_the_image(self):
        image = 1. + self.rng.normal(0., 0.05, (100, 100))
        self.assertAlmostEqual(estimate_noise(3. * image) /
                               estimate_noise(image), 3.)

    def test_bad_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            estimate_noise(np.ones(10))
        with self.assertRaises(ValueError):
            estimate_noise(np.ones((2, 50)))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
