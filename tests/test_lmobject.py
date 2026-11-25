import unittest

import numpy as np
import pandas as pd

from pylorenzmie.lib import LMObject
from pylorenzmie.theory import Instrument, Sphere


class Plain(LMObject):
    '''An object that is not a dataclass, so that it has the methods of
    ``LMObject`` itself.'''

    def __init__(self, value=1.):
        self.value = value

    @LMObject.properties.getter
    def properties(self):
        return dict(value=self.value)


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


class TestLMObjectBase(unittest.TestCase):
    '''The methods of ``LMObject`` that a dataclass replaces or overrides.'''

    def test_equality_compares_the_properties(self):
        self.assertEqual(Plain(2.), Plain(2.))
        self.assertNotEqual(Plain(2.), Plain(3.))

    def test_an_object_of_another_class_is_not_equal(self):
        self.assertNotEqual(Plain(), Instrument())
        self.assertIs(Plain().__eq__(1), NotImplemented)

    def test_numpy_scalars_are_serialized(self):
        obj = Plain(np.float32(2.5))
        other = Plain()
        other.from_json(obj.to_json())
        self.assertEqual(other.value, 2.5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
