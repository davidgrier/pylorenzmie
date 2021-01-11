import unittest

from fitting import Parameter


class TestParameter(unittest.TestCase):

    def setUp(self):
        self.parameter = Parameter()

    def test_vary(self):
        self.parameter.vary = True
        self.assertTrue(self.parameter.vary)
        self.parameter.vary = False
        self.assertFalse(self.parameter.vary)

    def test_value(self):
        value = 42.
        self.parameter.value = value
        self.assertEqual(self.parameter.value, value)

    def test_uncertainty(self):
        value = 42.
        self.parameter.uncertainty = value
        self.assertEqual(self.parameter.uncertainty, value)

if __name__ == '__main__':
    unittest.main()
