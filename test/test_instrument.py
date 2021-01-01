import unittest

from theory.Instrument import Instrument


class TestSphere(unittest.TestCase):

    def setUp(self):
        self.instrument = Instrument()

    def test_setmag(self):
        value = 0.120
        self.instrument.magnification = value
        self.assertEqual(self.instrument.magnification, value)

    def test_setnm(self):
        value = 1.34
        self.instrument.n_m = value
        self.assertEqual(self.instrument.n_m, value)

    def test_setwv(self):
        value = 0.447
        self.instrument.wavelength = value
        self.assertEqual(self.instrument.wavelength, value)        
        

if __name__ == '__main__':
    unittest.main()
