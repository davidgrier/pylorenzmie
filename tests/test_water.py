import unittest

import numpy as np

from pylorenzmie.utilities import density, refractiveindex


class TestWater(unittest.TestCase):
    '''Reference values are from the IAPWS-95 equation of state at
    atmospheric pressure, and from the IAPWS release on the refractive
    index of water (1997).'''

    def test_density_against_iapws95(self):
        # temperature [celsius]: density [kg/m^3]
        reference = {1: 999.902, 4: 999.975, 10: 999.702, 20: 998.207,
                     25: 997.048, 40: 992.216, 60: 983.196, 80: 971.790}
        for temperature, rho in reference.items():
            with self.subTest(temperature=temperature):
                self.assertAlmostEqual(density(temperature), rho, delta=0.01)

    def test_density_is_greatest_near_four_degrees(self):
        self.assertGreater(density(4.), density(0.))
        self.assertGreater(density(4.), density(10.))

    def test_density_decreases_on_heating_above_four_degrees(self):
        t = np.arange(5., 90., 5.)
        self.assertTrue(np.all(np.diff([density(x) for x in t]) < 0.))

    def test_refractive_index_check_value_of_the_iapws_release(self):
        # 0.2265 um and 25 C: n = 1.39277824 for the density 997.047435 that
        # the release uses; ours is 0.002 kg/m^3 different
        self.assertAlmostEqual(refractiveindex(0.2265, 25.), 1.39277824,
                               delta=2e-6)

    def test_refractive_index_at_the_sodium_line(self):
        self.assertAlmostEqual(refractiveindex(0.5893, 20.), 1.33335,
                               delta=2e-5)

    def test_dispersion(self):
        # blue light is refracted more than red
        blue, green, red = (refractiveindex(w, 24.) for w in (0.447, 0.532,
                                                              0.633))
        self.assertGreater(blue, green)
        self.assertGreater(green, red)

    def test_the_index_falls_as_the_water_warms(self):
        self.assertGreater(refractiveindex(0.589, 10.),
                           refractiveindex(0.589, 40.))

    def test_arrays_of_wavelengths(self):
        wavelength = np.array([0.4, 0.5, 0.6, 0.7])
        n = refractiveindex(wavelength, 24.)
        self.assertEqual(n.shape, wavelength.shape)
        self.assertTrue(np.all(np.diff(n) < 0.))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
