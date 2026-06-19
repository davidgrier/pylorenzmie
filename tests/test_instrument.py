import unittest
import numpy as np
from pylorenzmie.theory import Instrument


class TestInstrument(unittest.TestCase):

    def setUp(self):
        self.instrument = Instrument()

    # --- default values ---

    def test_default_wavelength(self):
        self.assertAlmostEqual(self.instrument.wavelength, 0.447)

    def test_default_magnification(self):
        self.assertAlmostEqual(self.instrument.magnification, 0.048)

    def test_default_numerical_aperture(self):
        self.assertAlmostEqual(self.instrument.numerical_aperture, 1.45)

    def test_default_noise(self):
        self.assertAlmostEqual(self.instrument.noise, 0.05)

    def test_default_n_m(self):
        self.assertAlmostEqual(self.instrument.n_m, 1.340)

    # --- attribute setters ---

    def test_set_wavelength(self):
        self.instrument.wavelength = 0.532
        self.assertAlmostEqual(self.instrument.wavelength, 0.532)

    def test_set_magnification(self):
        self.instrument.magnification = 0.120
        self.assertAlmostEqual(self.instrument.magnification, 0.120)

    def test_set_numerical_aperture(self):
        self.instrument.numerical_aperture = 1.20
        self.assertAlmostEqual(self.instrument.numerical_aperture, 1.20)

    def test_set_noise(self):
        self.instrument.noise = 0.1
        self.assertAlmostEqual(self.instrument.noise, 0.1)

    def test_set_n_m(self):
        self.instrument.n_m = 1.34
        self.assertAlmostEqual(self.instrument.n_m, 1.34)

    # --- properties protocol ---

    def test_properties_keys(self):
        props = self.instrument.properties
        for key in ('wavelength', 'magnification',
                    'numerical_aperture', 'noise', 'n_m'):
            self.assertIn(key, props)

    def test_properties_roundtrip(self):
        self.instrument.n_m = 1.339
        props = self.instrument.properties
        b = Instrument()
        b.properties = props
        self.assertAlmostEqual(b.n_m, self.instrument.n_m)

    # --- serialization ---

    def test_json_roundtrip(self):
        self.instrument.n_m = 1.341
        b = Instrument()
        b.from_json(self.instrument.to_json())
        self.assertAlmostEqual(b.n_m, self.instrument.n_m)

    def test_pandas_roundtrip(self):
        self.instrument.n_m = 1.341
        b = Instrument()
        b.from_pandas(self.instrument.to_pandas())
        self.assertAlmostEqual(b.n_m, self.instrument.n_m)

    # --- wavenumber ---

    def test_wavenumber_default(self):
        k = self.instrument.wavenumber()
        expected = (2. * np.pi / self.instrument.wavelength *
                    self.instrument.n_m *
                    self.instrument.magnification)
        self.assertAlmostEqual(k, expected)

    def test_wavenumber_in_vacuum(self):
        k = self.instrument.wavenumber(in_medium=False)
        expected = (2. * np.pi / self.instrument.wavelength *
                    self.instrument.magnification)
        self.assertAlmostEqual(k, expected)

    def test_wavenumber_unscaled(self):
        k = self.instrument.wavenumber(scaled=False)
        expected = (2. * np.pi / self.instrument.wavelength *
                    self.instrument.n_m)
        self.assertAlmostEqual(k, expected)

    def test_wavenumber_vacuum_unscaled(self):
        k = self.instrument.wavenumber(in_medium=False, scaled=False)
        expected = 2. * np.pi / self.instrument.wavelength
        self.assertAlmostEqual(k, expected)

    # --- repr ---

    def test_repr(self):
        self.assertIsInstance(repr(self.instrument), str)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
