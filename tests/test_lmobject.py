import unittest

import numpy as np
import pandas as pd

from pylorenzmie.theory import Instrument, Sphere


class TestLMObject(unittest.TestCase):
    '''The behavior that every pylorenzmie object shares.'''

    def test_equal_objects(self):
        self.assertEqual(Instrument(), Instrument())

    def test_objects_with_different_properties_differ(self):
        self.assertNotEqual(Instrument(n_m=1.33), Instrument(n_m=1.34))

    def test_objects_of_different_classes_differ(self):
        self.assertNotEqual(Instrument(), Sphere())

    def test_objects_are_not_hashable(self):
        with self.assertRaises(TypeError):
            hash(Instrument())

    def test_unknown_properties_are_ignored_and_logged(self):
        instrument = Instrument()
        with self.assertLogs(level='DEBUG') as logs:
            instrument.properties = {'no_such_property': 1., 'n_m': 1.5}
        self.assertEqual(instrument.n_m, 1.5)
        self.assertTrue(any('no_such_property' in m for m in logs.output))

    def test_json_round_trip(self):
        sphere = Sphere(a_p=0.8, n_p=1.4, x_p=3., y_p=4., z_p=150.)
        text = sphere.to_json()
        other = Sphere()
        other.from_json(text)
        self.assertEqual(sphere, other)

    def test_json_handles_numpy_scalars(self):
        sphere = Sphere()
        sphere.a_p = np.float64(0.9)
        sphere.z_p = np.float64(200.)
        other = Sphere()
        other.from_json(sphere.to_json())
        self.assertAlmostEqual(other.a_p, 0.9)

    def test_pandas_round_trip(self):
        sphere = Sphere(a_p=0.8, n_p=1.4, x_p=3., y_p=4., z_p=150.)
        series = sphere.to_pandas()
        self.assertIsInstance(series, pd.Series)
        other = Sphere()
        other.from_pandas(series)
        self.assertEqual(sphere, other)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
